
import time
import random
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple

import numpy as np

from BaseCache import BaseSimilarityCache  # your base class (observer, index factory, etc.)

def _ensure_f32_unit(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Assume input is already normalized; only enforce dtype/shape and avoid divide-by-zero."""
    v = np.asarray(x, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    if n == 0.0:
        return v
    # If already normalized, skip division (tolerate small drift)
    if abs(n - 1.0) <= 5e-3:
        return v
    return (v / n).astype(np.float32, copy=False)

def _cosine(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.dot(x, y))

def _ca_from_sim(sim: float) -> float:
    """Cosine-distance scaled to [0,1]: Ca = (1 - sim)/2"""
    return max(0.0, min(1.0, (1.0 - float(sim)) * 0.5))


@dataclass
class DuelState:
    opponent_key: str          # y (in cache)
    candidate_key: str         # y' (not in cache yet)
    candidate_emb: np.ndarray       # normalized float32 embedding of y'
    counter_y: float = 0.0     # cumulative savings if we keep y
    counter_yprime: float = 0.0# cumulative savings if we replace y with y'
    start_step: int = 0        # logical step when duel started
    last_update_step: int = 0  # last step we updated
    created_at: float = 0.0    # wall time (optional)

    def diff(self) -> float:
        return self.counter_yprime - self.counter_y


class DuelCache(BaseSimilarityCache):
    """
    Duel policy (Neglia et al.), adapted to continuous embedding space (CLIP) and FAISS Flat index.
    - Unaware: no lambda_freq required.
    - Maintains small set of concurrent 'duels' between a cached item y and a candidate y' (missed request).
    - For each incoming request r, we award 'savings' to y and y' and stop when |diff| >= delta or a timeout tau fires.
    - If y' wins, we replace y with y'.
    """

    def __init__(
        self,
        capacity: int,
        threshold: float,
        dim: int,
        backend: str,
        *,
        beta: float = 0.75,               # prob to match y as NN(y')
        delta: float = 0.05,              # stopping threshold on counters (same scale as Ca in [0,1])
        tau: int = 200,                   # timeout in number of requests processed
        interference_radius: float = 0.03,# if candidate embeddings are closer than this Ca, skip starting a new duel
        max_active_duels: int = 8,        # cap to keep per-request work bounded
        k_duel: int = 8,                  # how many neighbors to fetch per request for counter updates
        adaptive_thresh: bool = False,
        rng_seed: Optional[int] = None,
    ):
        super().__init__(capacity, threshold, dim, backend, adaptive_thresh)
        self.beta = float(beta)
        self.delta = float(delta)
        self.tau = int(tau)
        self.r_intf = float(interference_radius)
        self.max_active_duels = int(max_active_duels)
        self.k_duel = int(k_duel)
        self._duels: Dict[str, DuelState] = {}  # key by candidate_key (y')
        self._t = 0
        self._rng = random.Random(rng_seed)

        # Sanity: require backend to expose query_topk
        if not hasattr(self.index, "query_topk"):
            raise RuntimeError("DuelCache requires index.query_topk(emb, k). "
                               "Update Backend to expose query_topk returning List[(key, sim)].")

    # ---- Public API ----
    def query(self, key: str, emb: np.ndarray) -> bool:
        """Serve a request and update duels; possibly start/finish a duel on a miss."""
        self._t += 1
        x = _ensure_f32_unit(emb, 1e-12)

        # Serve request: approx-hit if best sim >= threshold
        best = self.index.query_topk(x, 1)
        hit = False
        if best:
            best_key, best_sim = best[0]
            if best_sim >= self.threshold:
                self._on_hit(best_key)
                self._notify('hit', best_key, best_sim)
                hit = True

        if not hit:
            # miss path
            self._notify('miss', key, best[0][1] if best else -1.0)

            if len(self.index.keys) < self.capacity:
                # Admit directly if space
                self.index.add(key, x)
                self._notify('add', key)
                self._on_add(key)
            else:
                # Capacity full: update all duels with this request then try to start/continue a duel for this candidate
                self._update_all_duels_with_request(x)

                if key in self._duels:
                    # Continue an existing duel for this candidate
                    self._maybe_finish_duel(self._duels[key])
                else:
                    self._maybe_start_duel(candidate_key=key, candidate_emb=x)

        # Even on hits, update duels (they learn from all requests)
        if hit:
            self._update_all_duels_with_request(x)
            # finishing condition may be met
            self._finish_expired_duels()

        return hit

    # ---- Duel mechanics ----
    def _update_all_duels_with_request(self, r: np.ndarray) -> None:
        """Update counters for each active duel using incoming request r."""
        if not self._duels:
            return

        # Pull top-k neighbors once; we will reuse to compute costs excluding various y's approximately.
        k = min(self.k_duel, max(1, len(self.index.keys)))
        topk = self.index.query_topk(r, k)  # List[(key, sim)] sorted desc by sim

        # Precompute baseline cost with current cache (y' not in cache)
        base_sim = topk[0][1] if topk else -1.0
        base_cost = min(1.0, _ca_from_sim(base_sim))  # Cr is 1.0 in normalized units

        for d in list(self._duels.values()):
            # Opponent y must still be in cache
            e_y = self.index.get_embedding(d.opponent_key)
            if e_y is None:
                # Opponent evicted by some external action; cancel duel
                self._notify('duel_cancel', {
                    'candidate': d.candidate_key,
                    'opponent': d.opponent_key,
                    'reason': 'opponent_missing'
                })
                self._duels.pop(d.candidate_key, None)
                continue

            # Candidate y' embedding kept inside duel
            e_yprime = d.candidate_emb

            # Compute c_without_y using topk excluding y (approximate via list scan)
            sim_alt = -1.0
            for k_y, s in topk:
                if k_y != d.opponent_key:
                    sim_alt = s
                    break
            c_without_y = min(1.0, _ca_from_sim(sim_alt))

            # Savings if we keep y and y is the best approximator for r
            sim_y = _cosine(r, e_y)
            ca_y = _ca_from_sim(sim_y)
            if (sim_y >= self.threshold) and (ca_y <= base_cost):
                d.counter_y += (c_without_y - ca_y)

            # Savings if we replace y by y' and y' is the best approximator for r
            sim_yprime = _cosine(r, e_yprime)
            ca_yprime = _ca_from_sim(sim_yprime)
            # For y' we compare with current baseline (since y' not in S yet)
            if (sim_yprime >= self.threshold) and (ca_yprime <= base_cost):
                d.counter_yprime += (base_cost - ca_yprime)

            d.last_update_step = self._t
            self._notify('duel_update', {
                'opponent': d.opponent_key,
                'candidate': d.candidate_key,
                'counter_y': d.counter_y,
                'counter_yprime': d.counter_yprime,
                'diff': d.diff(),
            })

        # After updating all, check finish conditions
        self._finish_expired_duels()

    def _finish_expired_duels(self) -> None:
        for cand_key, d in list(self._duels.items()):
            if abs(d.diff()) >= self.delta or (self._t - d.start_step) >= self.tau:
                self._maybe_finish_duel(d)

    def _maybe_finish_duel(self, d: DuelState) -> None:
        """If stopping condition met, apply replacement if candidate wins, then remove duel state."""
        if not (abs(d.diff()) >= self.delta or (self._t - d.start_step) >= self.tau):
            return
        winner = 'candidate' if d.diff() > 0 else 'opponent'
        if winner == 'candidate':
            # Replace y with y'
            self.index.remove(d.opponent_key)
            self._notify('evict', d.opponent_key)
            self._on_evict(d.opponent_key)
            self.index.add(d.candidate_key, d.candidate_emb)
            self._notify('add', d.candidate_key)
            self._on_add(d.candidate_key)
            self._notify('duel_finish', {'winner': 'candidate',
                                         'opponent': d.opponent_key,
                                         'candidate': d.candidate_key,
                                         'diff': d.diff()})
        else:
            self._notify('duel_finish', {'winner': 'opponent',
                                         'opponent': d.opponent_key,
                                         'candidate': d.candidate_key,
                                         'diff': d.diff()})
        self._duels.pop(d.candidate_key, None)

    def _maybe_start_duel(self, candidate_key: str, candidate_emb: np.ndarray) -> None:
        """Start a duel y (in cache) vs y' (candidate) if interference and cap allow."""
        if len(self._duels) >= self.max_active_duels:
            return

        # Interference: skip if candidate too close to an existing candidate
        for d in self._duels.values():
            if _ca_from_sim(_cosine(candidate_emb, d.candidate_emb)) <= self.r_intf:
                return

        # Choose opponent y: NN with prob beta, else uniform
        y_key = None
        if self._rng.random() < self.beta:
            nn = self.index.query_topk(candidate_emb, 1)
            if nn:
                y_key = nn[0][0]
        if y_key is None:
            keys = self.index.keys
            if keys:
                y_key = self._rng.choice(keys)

        if y_key is None:
            return  # nothing to duel with

        self._duels[candidate_key] = DuelState(
            opponent_key=y_key,
            candidate_key=candidate_key,
            candidate_emb=candidate_emb,
            start_step=self._t,
            last_update_step=self._t,
            created_at=time.time()
        )
        self._notify('duel_start', {'opponent': y_key, 'candidate': candidate_key})
        # No immediate replacement; counters will accrue on future requests

    # ---- Hooks for observer in BaseSimilarityCache ----
    def _on_add(self, key: str): pass
    def _on_evict(self, key: str): pass
    def _on_hit(self, key: str): pass
