from typing import Dict, Callable, Optional, List
from dataclasses import dataclass
import numpy as np

from BaseCache import BaseSimilarityCache

def _ensure_f32_unit(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(x, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    if n <= eps:
        return v
    if abs(n - 1.0) <= 5e-3:
        return v
    return (v / n).astype(np.float32, copy=False)

def _cosine(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.dot(x, y))

def _ca_from_sim(sim: float) -> float:
    return max(0.0, min(1.0, (1.0 - float(sim)) * 0.5))


@dataclass
class GreedyConfig:
    cr: float = 1.0
    sample_size: Optional[int] = 256
    rng_seed: Optional[int] = None


class GreedyCache(BaseSimilarityCache):
    """
    Greedy (λ-aware) per Neglia et al.
    - Usa lambda_freq (stima delle frequenze di richiesta) per minimizzare il costo atteso.
    - Ca default: (1 - cos)/2 in [0,1]. Cr default: 1.0.
    - Non ricalcola embedding: li legge dall'indice o da embedding_lookup(key).
    """

    def __init__(
        self,
        capacity: int,
        threshold: float,
        dim: int,
        backend: str,
        *,
        lambda_freq: Dict[str, float],
        embedding_lookup: Optional[Callable[[str], Optional[np.ndarray]]] = None,
        ca_func: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
        config: Optional[GreedyConfig] = None,
        adaptive_thresh: bool = False,
    ):
        super().__init__(capacity, threshold, dim, backend, adaptive_thresh)
        self.ca = (lambda a,b: _ca_from_sim(_cosine(a,b))) if ca_func is None else ca_func
        self.cfg = config or GreedyConfig()
        self.embedding_lookup = embedding_lookup

        self.lambda_freq: Dict[str, float] = {}
        tot = 0.0
        for k, w in (lambda_freq or {}).items():
            v = float(w)
            if v > 0.0:
                self.lambda_freq[k] = v
                tot += v
        if tot > 0.0:
            for k in list(self.lambda_freq.keys()):
                self.lambda_freq[k] /= tot
            self._degenerate = False
        else:
            self._degenerate = True

        self._lambda_emb_cache: Dict[str, np.ndarray] = {}
        self._rng = np.random.default_rng(self.cfg.rng_seed)

        if not hasattr(self.index, "get_embedding"):
            raise RuntimeError("GreedyCache richiede index.get_embedding(key).")

    def query(self, key: str, emb: np.ndarray) -> bool:
        x = _ensure_f32_unit(emb)
        best_key, best_sim = self.index.query(x, topk=1)
        if best_key is not None and best_sim >= self.threshold:
            self._on_hit(best_key)
            self._notify('hit', best_key, best_sim)
            return True

        self._notify('miss', key, best_sim if best_key is not None else None)

        if len(self.index.keys) < self.capacity:
            self.index.add(key, x)
            self._notify('add', key, None)
            self._on_add(key)
            return False

        if self._degenerate or not self.lambda_freq:
            victim = self._worst_sim_vs_query(x)
            if victim is not None:
                self.index.remove(victim)
                self._notify('evict', victim, None)
                self._on_evict(victim)
                self.index.add(key, x)
                self._notify('add', key, None)
                self._on_add(key)
            return False

        current_keys = list(self.index.keys)
        current_cost = self._expected_cost(current_keys)

        best_impr = 0.0
        best_victim = None
        for victim in current_keys:
            new_cost = self._expected_cost_replace(current_keys, victim, key, x)
            impr = current_cost - new_cost
            if impr > best_impr:
                best_impr = impr
                best_victim = victim

        if best_victim is not None and best_impr > 0.0:
            self.index.remove(best_victim)
            self._notify('evict', best_victim, None)
            self._on_evict(best_victim)
            self.index.add(key, x)
            self._notify('add', key, None)
            self._on_add(key)

        return False

    # ----- expectation helpers -----
    def _expected_cost(self, keys: List[str]) -> float:
        if not keys:
            return self.cfg.cr

        cache_vecs = []
        for k in keys:
            v = self.index.get_embedding(k)
            if v is not None:
                cache_vecs.append(v)
        if not cache_vecs:
            return self.cfg.cr

        items = list(self.lambda_freq.items())
        if self.cfg.sample_size is not None and self.cfg.sample_size < len(items):
            lam_keys = np.array([k for k,_ in items], dtype=object)
            lam_probs = np.array([w for _,w in items], dtype=np.float64)
            lam_probs = lam_probs / lam_probs.sum()
            idx = self._rng.choice(len(lam_keys), size=self.cfg.sample_size, replace=False, p=lam_probs)
            items = [(lam_keys[i].item(), float(lam_probs[i])) for i in idx]

        total = 0.0
        ws = 0.0
        for xk, w in items:
            ex = self._lambda_embedding(xk)
            if ex is None:
                continue
            best_ca = self.cfg.cr
            for ey in cache_vecs:
                s = _cosine(ex, ey)
                if s >= self.threshold:
                    c = self.ca(ex, ey)
                    if c < best_ca:
                        best_ca = c
            total += w * min(self.cfg.cr, best_ca)
            ws += w

        return float(total / ws) if ws > 0 else self.cfg.cr

    def _expected_cost_replace(self, current_keys: List[str], victim: str, cand_key: str, cand_vec: np.ndarray) -> float:
        keys2 = [k for k in current_keys if k != victim]
        keys2.append(cand_key)
        return self._expected_cost_with_injected(keys2, cand_key, cand_vec)

    def _expected_cost_with_injected(self, keys: List[str], injected_key: str, injected_vec: np.ndarray) -> float:
        cache_vecs = []
        for k in keys:
            if k == injected_key:
                cache_vecs.append(injected_vec)
            else:
                v = self.index.get_embedding(k)
                if v is not None:
                    cache_vecs.append(v)
        if not cache_vecs:
            return self.cfg.cr

        items = list(self.lambda_freq.items())
        if self.cfg.sample_size is not None and self.cfg.sample_size < len(items):
            lam_keys = np.array([k for k,_ in items], dtype=object)
            lam_probs = np.array([w for _,w in items], dtype=np.float64)
            lam_probs = lam_probs / lam_probs.sum()
            idx = self._rng.choice(len(lam_keys), size=self.cfg.sample_size, replace=False, p=lam_probs)
            items = [(lam_keys[i].item(), float(lam_probs[i])) for i in idx]

        total = 0.0
        ws = 0.0
        for xk, w in items:
            ex = self._lambda_embedding(xk)
            if ex is None:
                continue
            best_ca = self.cfg.cr
            for ey in cache_vecs:
                s = _cosine(ex, ey)
                if s >= self.threshold:
                    c = self.ca(ex, ey)
                    if c < best_ca:
                        best_ca = c
            total += w * min(self.cfg.cr, best_ca)
            ws += w

        return float(total / ws) if ws > 0 else self.cfg.cr

    def _lambda_embedding(self, key: str) -> Optional[np.ndarray]:
        v = self._lambda_emb_cache.get(key)
        if v is not None:
            return v
        if self.embedding_lookup is not None:
            vv = self.embedding_lookup(key)
            if vv is not None:
                vv = _ensure_f32_unit(vv)
                self._lambda_emb_cache[key] = vv
                return vv
        vv = self.index.get_embedding(key)
        if vv is not None:
            vv = _ensure_f32_unit(vv)
            self._lambda_emb_cache[key] = vv
            return vv
        return None

    def _worst_sim_vs_query(self, q: np.ndarray) -> Optional[str]:
        worst_key = None
        worst_sim = 1e9
        for k in list(self.index.keys):
            ek = self.index.get_embedding(k)
            if ek is None:
                continue
            s = _cosine(q, ek)
            if s < worst_sim:
                worst_sim = s
                worst_key = k
        return worst_key

    def _on_add(self, key: str): pass
    def _on_evict(self, key: str): pass
    def _on_hit(self, key: str): pass
