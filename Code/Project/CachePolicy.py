import numpy as np
from typing import Optional
from collections import defaultdict
from collections import OrderedDict
from BaseCache import BaseSimilarityCache

class LRUCache(BaseSimilarityCache):
    """Least-Recently-Used eviction policy."""
    def __init__(self, capacity: int, threshold: float, dim: int, backend: str, adaptive_thresh = False):
        super().__init__(capacity, threshold, dim, backend, adaptive_thresh)
        from collections import OrderedDict
        self.order = OrderedDict()  # key -> None

    def _on_hit(self, key: str):
        # move to end = most recently used
        self.order.move_to_end(key)

    def _on_add(self, key: str):
        self.order[key] = None

    def _on_evict(self, key: str):
        del self.order[key]

    def _select_eviction(self) -> str:
        # least recently used = first key
        return next(iter(self.order))

class LFUCache(BaseSimilarityCache):
    """Least-Frequently-Used eviction policy."""
    def __init__(self, capacity: int, threshold: float, dim: int, backend: str, adaptive_thresh = False):
        super().__init__(capacity, threshold, dim, backend, adaptive_thresh)
        self.freq = defaultdict(int)

    def _on_hit(self, key: str):
        self.freq[key] += 1

    def _on_add(self, key: str):
        self.freq[key] = 1

    def _on_evict(self, key: str):
        del self.freq[key]

    def _select_eviction(self) -> str:
        # key with minimal freq
        return min(self.freq, key=lambda k: self.freq[k])

class TTLCache(BaseSimilarityCache):
    """Time-to-Live eviction: removes entries not seen in last ttl requests."""
    def __init__(
        self,
        capacity: int,
        threshold: float,
        dim: int,
        ttl: int,
        backend: str,
        adaptive_thresh: bool = False
    ):
        super().__init__(capacity, threshold, dim, backend, adaptive_thresh)
        self.ttl = ttl
        self.time = 0
        self.last_seen = {}  # key -> last request time

    def query(self, key: str, emb: np.ndarray) -> bool:
        # expire old entries first
        expired = [k for k, t in self.last_seen.items() if self.time - t > self.ttl]
        for k in expired:
            self.index.remove(k)
            del self.last_seen[k]
        hit = super().query(key, emb)
        # update time
        self.time += 1
        return hit

    def _on_hit(self, key: str):
        self.last_seen[key] = self.time

    def _on_add(self, key: str):
        self.last_seen[key] = self.time

    def _select_eviction(self) -> str:
        # not used: eviction on miss picks oldest by time
        return min(self.last_seen, key=lambda k: self.last_seen[k])
    


#   Randomized Cache Policies   #

class RNDLRUCache(LRUCache):
    """Randomized LRU where threshold varies at each query."""
    def __init__(self, capacity, threshold, dim, backend=True, min_threshold=0.5, max_threshold=1.0):

        super().__init__(capacity, threshold, dim, backend)
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

    def _should_accept(self, key: Optional[str], similarity: float) -> bool:
        # threshold random per ogni richiesta
        self.threshold = np.random.uniform(self.min_threshold, self.max_threshold)
        return similarity >= self.threshold


class RNDTTLCache(TTLCache):
    """Randomized TTL with varying similarity threshold per request."""
    def __init__(self, capacity, threshold, dim, ttl, backend=True, min_threshold=0.5, max_threshold=1.0):
        super().__init__(capacity, threshold, dim, ttl, backend)
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

    def _should_accept(self, key: Optional[str], similarity: float) -> bool:
        # threshold random per ogni richiesta
        self.threshold = np.random.uniform(self.min_threshold, self.max_threshold)
        return similarity >= self.threshold


class TwoLRUCache(LRUCache):
    """
    TwoLRU: filtro LRU + cache principale.
    Promuove un oggetto solo se è nel filtro.
    """
    def __init__(self, capacity: int, threshold: float, dim: int, backend: str,
                 filter_ratio: float = 0.1, adaptive_thresh: bool = False):
        super().__init__(capacity, threshold, dim, backend, adaptive_thresh)
        self.filter_size = max(1, int(capacity * filter_ratio))
        self.filter_order = OrderedDict()

    @BaseSimilarityCache.adaptive_acceptance
    def _should_accept(self, key, sim) -> bool:
        # Se simile a un oggetto già in cache → hit diretto
        if sim >= self.threshold:
            return True
        # Altrimenti, controlla se è nel filtro
        if key in self.filter_order:
            self.filter_order.pop(key)  # promuovo → tolgo dal filtro
            return True
        else:
            # Prima volta: inserisco nel filtro, rifiuto
            self.filter_order[key] = None
            if len(self.filter_order) > self.filter_size:
                self.filter_order.popitem(last=False)
            return False
