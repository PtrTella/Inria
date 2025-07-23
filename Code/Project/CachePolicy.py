import BaseCache as bc
import numpy as np
from collections import defaultdict

class LRUCache(bc.BaseSimilarityCache):
    """Least-Recently-Used eviction policy."""
    def __init__(self, capacity: int, threshold: float, dim: int, use_faiss: bool = True):
        super().__init__(capacity, threshold, dim, use_faiss)
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

class LFUCache(bc.BaseSimilarityCache):
    """Least-Frequently-Used eviction policy."""
    def __init__(self, capacity: int, threshold: float, dim: int, use_faiss: bool = True):
        super().__init__(capacity, threshold, dim, use_faiss)
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

class TTLCache(bc.BaseSimilarityCache):
    """Time-to-Live eviction: removes entries not seen in last ttl requests."""
    def __init__(
        self,
        capacity: int,
        threshold: float,
        dim: int,
        ttl: int,
        use_faiss: bool = True
    ):
        super().__init__(capacity, threshold, dim, use_faiss)
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
    

class RNDLRUCache(LRUCache):
    """Randomized LRU where threshold varies at each query."""
    def __init__(self, capacity, threshold, dim, use_faiss=True, min_threshold=0.5, max_threshold=1.0):
        super().__init__(capacity, threshold, dim, use_faiss)
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

    def query(self, key: str, emb: np.ndarray) -> bool:
        # random threshold every time
        self.threshold = np.random.uniform(self.min_threshold, self.max_threshold)
        return super().query(key, emb)


class RNDTTLCache(TTLCache):
    """Randomized TTL with varying similarity threshold per request."""
    def __init__(self, capacity, threshold, dim, ttl, use_faiss=True, min_threshold=0.5, max_threshold=1.0):
        super().__init__(capacity, threshold, dim, ttl, use_faiss)
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

    def query(self, key: str, emb: np.ndarray) -> bool:
        expired = [k for k, t in self.last_seen.items() if self.time - t > self.ttl]
        for k in expired:
            self.index.remove(k)
            del self.last_seen[k]
        # random threshold
        self.threshold = np.random.uniform(self.min_threshold, self.max_threshold)
        hit = bc.BaseSimilarityCache.query(self, key, emb)
        self.time += 1
        return hit

