
import random
from typing import Dict, Callable, Optional, List, Tuple
import numpy as np

from BaseCache import BaseSimilarityCache

Array = np.ndarray

def _ensure_f32(x: Array, dim: int) -> Array:
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    if arr.shape[0] != dim:
        raise ValueError(f"Embedding must be shape ({dim},), got {arr.shape}")
    return arr

def _l2norm(x: Array, eps: float = 1e-12) -> Array:
    n = np.linalg.norm(x)
    if n < eps:
        return x.astype(np.float32, copy=True)
    return (x / n).astype(np.float32, copy=False)

def _cosine(x: Array, y: Array) -> float:
    return float(np.dot(x, y))

def _ca_cosine_distance(x: Array, y: Array) -> float:
    # On normalized vectors, cosine distance = 1 - cosine sim
    return 1.0 - _cosine(x, y)


class GreedyCache(BaseSimilarityCache):
    """
    Greedy similarity cache (cost-aware).
    - Keeps item only if swapping reduces the expected cost over the (known) request distribution λ(x).
    - Requires: lambda_freq: Dict[key, λ(x)]  (non-negative weights)
    - Needs a cost of approximation ca_func(x, y) and a miss cost cr.
    Expected cost model for a request x given a set S of cached keys:
        cost(x; S) =  min_{y in S with sim(x,y) >= threshold}  Ca(x, y)   (if at least one meets threshold)
                      otherwise: Cr
    We evaluate improvement when swapping a victim with the candidate (key, emb).
    """

    def __init__(
        self,
        capacity: int,
        threshold: float,
        dim: int,
        backend: str,
        lambda_freq: Dict[str, float],
        ca_func: Callable[[Array, Array], float] = _ca_cosine_distance,
        cr: float = 1.0,
        adaptive_thresh: bool = False,
    ):
        super().__init__(capacity, threshold, dim, backend, adaptive_thresh)
        self.lambda_freq = dict(lambda_freq or {})
        self.ca_func = ca_func
        self.cr = float(cr)

        if not self.lambda_freq:
            # No distribution ⇒ Greedy cannot estimate anything ⇒ degrade to "always admit if space else replace worst-sim"
            # but we keep a flag to do the simplified heuristic.
            self._degenerate = True
        else:
            # normalize weights (not required but keeps numbers stable)
            s = sum(max(0.0, float(v)) for v in self.lambda_freq.values()) or 1.0
            for k in list(self.lambda_freq.keys()):
                self.lambda_freq[k] = max(0.0, float(self.lambda_freq[k])) / s
            self._degenerate = False

    # ---- Helper: expected cost given a set of keys ----
    def _expected_cost(self, keys: Optional[List[str]] = None, inject: Optional[Tuple[str, Array]] = None) -> float:
        """
        Compute E[cost] over lambda_freq for a given set of cached keys.
        - keys: explicit set (defaults to current self.index.keys)
        - inject: (key, emb) to treat as included (emb must be normalized, float32)
        """
        if self._degenerate or not self.lambda_freq:
            return 0.0  # not used in degenerate path

        if keys is None:
            keys = list(self.index.keys)

        # Build lookup: key -> emb (normalized). If inject is provided, override/add it.
        emb_of: Dict[str, Array] = {}
        for k in keys:
            ve = self.index.get_embedding(k)
            if ve is not None:
                emb_of[k] = ve.astype(np.float32, copy=False)
        if inject is not None:
            k_inj, e_inj = inject
            emb_of[k_inj] = _l2norm(_ensure_f32(e_inj, self.dim))

        # Pre-materialize a list of (key, emb) for fast scans
        cached_items = list(emb_of.items())

        # For each request key x with λ(x) > 0, we need the embedding of x as well.
        # If x is itself in emb_of, we can use that. Otherwise, try to obtain from index.get_embedding(x);
        # if not available, we skip it (assume unaffected by current cache composition).
        expected = 0.0
        for x, lam in self.lambda_freq.items():
            if lam <= 0.0:
                continue
            ex = emb_of.get(x)
            if ex is None:
                ex = self.index.get_embedding(x)  # might be None if x is not cached and unknown to index
            if ex is None:
                # No embedding available → we cannot evaluate; ignore contribution (assume constant baseline)
                continue

            # compute best cost for x given the cached set
            best_cost = self.cr
            for y, ey in cached_items:
                sim = _cosine(ex, ey)
                if sim >= self.threshold:
                    cxy = self.ca_func(ex, ey)
                    if cxy < best_cost:
                        best_cost = cxy
                        # optional: early exit if zero
                        if best_cost <= 0.0:
                            break
            expected += lam * best_cost
        return float(expected)

    def query(self, key: str, emb: Array) -> bool:
        # Normalize input
        emb = _l2norm(_ensure_f32(emb, self.dim))

        # 1) Try to answer from cache (approximate hit if similarity >= threshold)
        best_key, best_sim = self.index.query(emb, topk=1)
        if best_key is not None and best_sim >= self.threshold:
            self._on_hit(best_key)
            self._notify('hit', best_key, best_sim)
            return True

        # 2) Miss: decide admission by greedy improvement
        if len(self.index.keys) < self.capacity:
            # free slot: admit directly
            self.index.add(key, emb)
            self._notify('add', key)
            self._on_add(key)
            self._notify('miss', key, best_sim)
            return False

        # Full cache: evaluate greedy swap
        if self._degenerate:
            # Heuristic: evict the item with the worst similarity to the query (or random)
            # Compute sims for all cached items to the query and drop the min
            sims = []
            for k in list(self.index.keys):
                ek = self.index.get_embedding(k)
                s = float(np.dot(emb, ek)) if ek is not None else -1.0
                sims.append((s, k))
            sims.sort()  # ascending → worst first
            victim = sims[0][1] if sims else next(iter(self.index.keys))
            self.index.remove(victim)
            self._notify('evict', victim)
            self._on_evict(victim)
            self.index.add(key, emb)
            self._notify('add', key)
            self._on_add(key)
            self._notify('miss', key, best_sim)
            return False

        # Proper greedy: compute improvement per possible victim
        current_keys = list(self.index.keys)
        current_cost = self._expected_cost(current_keys)

        best_improvement = 0.0
        best_victim: Optional[str] = None

        for victim in current_keys:
            temp_keys = [k for k in current_keys if k != victim]
            # cost if we inject the candidate instead of victim
            new_cost = self._expected_cost(temp_keys, inject=(key, emb))
            improvement = current_cost - new_cost
            if improvement > best_improvement:
                best_improvement = improvement
                best_victim = victim

        if best_improvement > 0.0 and best_victim is not None:
            self.index.remove(best_victim)
            self._notify('evict', best_victim)
            self._on_evict(best_victim)
            self.index.add(key, emb)
            self._notify('add', key)
            self._on_add(key)
        # else: do not admit

        self._notify('miss', key, best_sim)
        return False

    def _on_add(self, key: str): pass
    def _on_evict(self, key: str): pass
    def _on_hit(self, key: str): pass


class DuelCache(BaseSimilarityCache):
    """
    DUEL (policy dueling) wrapper.
    Runs two candidate policies (A, B) in parallel using shadow evaluation and
    routes real traffic through the currently best policy. Periodically re-evaluates.
    - policy_a_cls / policy_b_cls: subclasses of BaseSimilarityCache
    - Each policy gets half capacity by default (configurable via split).
    - We mirror queries to both (shadow) but only the active policy affects the shared index.
      (Implementation note: to keep things simple, we simulate both policies internally on their
       own private indexes created via the same backend factory.)
    """

    def __init__(
        self,
        capacity: int,
        threshold: float,
        dim: int,
        backend: str,
        policy_a_cls,
        policy_b_cls,
        policy_a_kwargs: Optional[dict] = None,
        policy_b_kwargs: Optional[dict] = None,
        eval_window: int = 200,
        split: Tuple[int, int] = (1, 1),
        adaptive_thresh: bool = False,
    ):
        super().__init__(capacity, threshold, dim, backend, adaptive_thresh)
        self.eval_window = int(eval_window)
        self.split = split
        self._history: List[Tuple[bool, bool]] = []  # (hitA, hitB) per query
        self._active = 'A'

        # Create shadow caches with their own internal indexes
        # We import inside to avoid circular imports if the user keeps files split.
        from Backend import create_similarity_index

        capA = max(1, capacity * split[0] // max(1, sum(split)))
        capB = max(1, capacity * split[1] // max(1, sum(split)))

        kwargsA = dict(policy_a_kwargs or {})
        kwargsB = dict(policy_b_kwargs or {})

        # Force backend for shadows (reuse same backend string)
        self._shadowA = policy_a_cls(capA, threshold, dim, backend, **kwargsA)
        self._shadowB = policy_b_cls(capB, threshold, dim, backend, **kwargsB)

        # Real index kept in the parent (self.index). Active policy will execute mutations on it.
        # Shadows maintain their own internal indexes. We won't try to keep them identical to the real cache;
        # dueling compares *rates* over the same stream, not exact states.

    def _switch_if_needed(self):
        # Evaluate last eval_window queries; choose policy with higher hit rate
        if len(self._history) < self.eval_window:
            return
        # Compute rolling window stats
        window = self._history[-self.eval_window:]
        hitA = sum(1 for a, _ in window if a)
        hitB = sum(1 for _, b in window if b)
        if hitA > hitB and self._active != 'A':
            self._active = 'A'
            self._notify('policy', 'activate_A')
        elif hitB > hitA and self._active != 'B':
            self._active = 'B'
            self._notify('policy', 'activate_B')

    def query(self, key: str, emb: Array) -> bool:
        # Mirror to both shadows
        hitA = self._shadowA.query(key, emb)
        hitB = self._shadowB.query(key, emb)
        self._history.append((hitA, hitB))
        self._switch_if_needed()

        # Execute on the active policy against the real index:
        # To keep the parent index authoritative, we re-run the active policy's decision,
        # but do it using the parent's index. We'll emulate by applying the same simple logic
        # used by most caches: check for hit; if miss, delegate admission decision to chosen policy.
        emb_n = _l2norm(_ensure_f32(emb, self.dim))
        best_key, best_sim = self.index.query(emb_n, topk=1)
        if best_key is not None and best_sim >= self.threshold:
            self._on_hit(best_key)
            self._notify('hit', best_key, best_sim)
            return True

        # Miss: ask the active shadow what it would do by calling its admission step again.
        # We need a hook; simplest is to temporarily swap self.index to reuse their logic.
        active = self._shadowA if self._active == 'A' else self._shadowB

        # Monkey-patch: run active.query against our real index by temporarily redirecting.
        real_index = self.index
        try:
            self.index = real_index  # already is
            return active.__class__.query(self, key, emb_n)  # call the class method using our self
        finally:
            self.index = real_index

    def _on_add(self, key: str): pass
    def _on_evict(self, key: str): pass
    def _on_hit(self, key: str): pass
