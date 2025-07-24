from functools import wraps
import numpy as np
from Backend import ISimilarityIndex, create_similarity_index
from typing import Callable, List, Tuple, Optional



class BaseSimilarityCache:
    """
    Base class for similarity caches. You can register observers
    that riceveranno eventi: 'hit', 'miss', 'add', 'evict'.
    """
    def __init__(
        self, capacity: int, threshold: float, dim: int, backend: str = "faiss", adaptive_thresh = False
    ):
        self.capacity = capacity
        self.threshold = threshold
        self.adaptive_thresh = adaptive_thresh
        self.index = create_similarity_index(dim, backend)
        self._observers: List[Callable[[str, str, "BaseSimilarityCache"], None]] = []

    def register_observer(self, fn: Callable[[str, str, 'BaseSimilarityCache'], None]):
        """fn(event_type, key, cache)"""
        self._observers.append(fn)

    def _notify(self, event_type: str, key: Optional[str], sim: Optional[float] = None):
        for fn in self._observers:
            fn(event_type, key, self, sim)

    def query(self, key: str, emb: np.ndarray) -> bool:
        best_key, best_sim = self.index.query(emb)

        if self._should_accept(best_key, best_sim):
            self._on_hit(best_key)
            self._notify('hit', best_key, best_sim)
            return True
        else:
            self._notify('miss', key, best_sim)
            self._on_miss(key, emb)
            return False

    def adaptive_acceptance(method):
        def wrapped(self, key, sim):
            if self.adaptive_thresh:
                occ = len(self.index.keys) / self.capacity
                dyn_thresh = min(1.0, self.threshold * (1 + self.adaptive_thresh * occ))
                self.threshold = dyn_thresh
            return method(self, key, sim)
        return wrapped

    @adaptive_acceptance
    def _should_accept(self, key: Optional[str], sim: float) -> bool:
        return sim >= self.threshold  # Default behaviour

    def _on_miss(self, key: str, emb: np.ndarray):
        if len(self.index.keys) >= self.capacity:
            ev = self._select_eviction()
            self.index.remove(ev)
            self._notify('evict', ev)
            self._on_evict(ev)
        self.index.add(key, emb)
        self._notify('add', key)
        self._on_add(key)

    # Subclasses must implement:
    def _on_add(self, key: str):        pass
    def _on_evict(self, key: str):      pass
    def _on_hit(self, key: str):        pass
    def _select_eviction(self) -> str:  pass
