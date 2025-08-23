import random
import time
from typing import Dict, Optional
import numpy as np
from collections import deque
from dataclasses import dataclass
import random
import numpy as np
from BaseCache import BaseSimilarityCache

def _ensure_unit(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(x, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    if n <= eps: return v
    if abs(n - 1.0) <= 5e-3: return v
    return (v / n).astype(np.float32, copy=False)

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))

def _ca_from_sim(sim: float) -> float:
    # Ca in [0,1] se i vettori sono normalizzati
    return max(0.0, min(1.0, (1.0 - float(sim)) * 0.5))

def _ca_from_sim_theta(sim: float, theta: float, eps: float = 1e-6) -> float:
    # costa 0 a sim=1; costa 1 al bordo sim=theta; >1 se sotto soglia (troncherà a 1 via min)
    den = max(eps, 1.0 - float(theta))
    return max(0.0, (1.0 - float(sim)) / den)


class QLRUDeltaCCache(BaseSimilarityCache):
    """
    qLRU-ΔC (Neglia et al.).
    - Coda LRU (deque) per l'ordine.
    - Miss: inserimento di x con prob. q.
    - Approx-hit: serve z=argmin Ca(x,y); refresh z con prob. (C(x,S\{z})-Ca(x,z))/Cr;
                  con prob. q*Ca(x,z)/Cr, recupera comunque x e lo inserisce (head).
    """
    def __init__(self, capacity, threshold, dim, backend,
                 ca_func=None, cr=1.0, q=0.2, adaptive_thresh=False,
                 mode='paper'):
        super().__init__(capacity, threshold, dim, backend, adaptive_thresh)
        # default: Ca=(1-cos)/2 coerente con indice coseno
        self.ca = (
            ca_func
            if ca_func is not None
            else (
                lambda sim: _ca_from_sim_theta(sim, self.threshold)
            )
        )
        self.cr = float(cr)
        self.q = float(q)
        self.queue = deque()     # LRU order: left = MRU, right = LRU
        self.mode = mode         # 'paper' usa Ca<=Cr; 'threshold' usa self.threshold

        # Sanity: per refresh serve 2-NN
        self._has_topk = hasattr(self.index, "query_topk")

    # --- helper per decisione approx-hit coerente
    def _admissible(self, best_sim: float, best_ca: float) -> bool:
        if self.mode == 'paper':
            return best_ca <= self.cr
        return (best_sim is not None) and (best_sim >= float(self.threshold))

    def query(self, key: str, emb: np.ndarray) -> bool:
        x = _ensure_unit(emb)

        # 1-NN (e idealmente 2-NN) per hit e refresh prob
        if self._has_topk:
            topk = self.index.query_topk(x, 2)  # [(key, sim), ...]
        else:
            ## NON SERVE IN TEORIA GESTITO GIA DAL BACKEND
            # fallback: scan lineare
            sims = []
            for k in self.index.keys:
                vk = self.index.get_embedding(k)
                if vk is not None:
                    sims.append((k, _cosine(x, vk)))
            sims.sort(key=lambda kv: kv[1], reverse=True)
            topk = sims[:2]

        hit = False
        best_key, best_sim = (None, None)
        if topk:
            best_key, best_sim = topk[0]
            if self.ca(best_sim) < self.cr:
                # ---- approx-hit path ----
                z = best_key
                ca_xz = self.ca(best_sim)

                # costo senza z: usa 2° vicino se esiste, altrimenti Cr
                if len(topk) >= 2:
                    _, sim2 = topk[1]
                    ca_alt = self.ca(sim2)
                    c_wo_z = min(self.cr, ca_alt)
                else:
                    c_wo_z = self.cr

                # refresh prob = (C(x,S\{z}) - Ca(x,z)) / Cr  (clamp [0,1])
                p_refresh = max(0.0, min(1.0, (c_wo_z - ca_xz) / self.cr))
                if random.random() < p_refresh:
                    # move-to-front
                    if z in self.queue:
                        self.queue.remove(z)
                    self.queue.appendleft(z)

                # con prob q*Ca/Cr recupero comunque x e lo inserisco (paper)
                p_insert_x = max(0.0, min(1.0, self.q * (ca_xz / self.cr)))
                if random.random() < p_insert_x:
                    self._insert(key, x)

                self._on_hit(z)
                self._notify('hit', z, best_sim)
                hit = True

        if hit:
            return True

        # ---- miss path (Ca > Cr): inserisci con prob. q ----
        if random.random() < self.q:
            self._insert(key, x)

        self._notify('miss', key, best_sim if best_sim is not None else None)
        return False

    def _insert(self, key: str, v: np.ndarray):
        # LRU insert at head
        if key in self.queue:
            self.queue.remove(key)
        self.queue.appendleft(key)

        self.index.add(key, v)
        self._notify('add', key, None)
        self._on_add(key)

        # Evict se oltre capacità
        if len(self.queue) > self.capacity:
            victim = self.queue.pop()
            self.index.remove(victim)
            self._notify('evict', victim, None)
            self._on_evict(victim)

    # Hooks (observer)
    def _on_add(self, key: str): pass
    def _on_evict(self, key: str): pass
    def _on_hit(self, key: str): pass


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

        self.cr = 1.0
        self.ca = lambda sim: _ca_from_sim_theta(sim, self.threshold)

        # Sanity: require backend to expose query_topk
        if not hasattr(self.index, "query_topk"):
            raise RuntimeError("DuelCache requires index.query_topk(emb, k). "
                               "Update Backend to expose query_topk returning List[(key, sim)].")

    # ---- Public API ----
    def query(self, key: str, emb: np.ndarray) -> bool:
        """Serve a request and update duels; possibly start/finish a duel on a miss."""
        self._t += 1
        x = _ensure_unit(emb, 1e-12)

        # Serve request: approx-hit if best sim >= threshold
        best = self.index.query_topk(x, 1)
        hit = False
        if best:
            best_key, best_sim = best[0]
            if self.ca(best_sim) < self.cr:
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
        base_cost = min(self.cr, self.ca(base_sim))

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
            c_without_y = min(self.cr, self.ca(sim_alt))

            # Savings if we keep y and y is the best approximator for r
            sim_y = _cosine(r, e_y)
            ca_y = self.ca(sim_y)
            if (sim_y >= self.threshold) and (ca_y <= base_cost):
                d.counter_y += (c_without_y - ca_y)

            # Savings if we replace y by y' and y' is the best approximator for r
            sim_yprime = _cosine(r, e_yprime)
            ca_yprime = self.ca(sim_yprime)
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
            if _ca_from_sim_theta(_cosine(candidate_emb, d.candidate_emb), self.threshold) <= self.r_intf:
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
