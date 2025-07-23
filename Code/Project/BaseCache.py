import numpy as np
import faiss
from typing import Callable, List, Tuple, Optional

class SimilarityIndex:
    """
    Wraps a FAISS inner-product index for fast nearest-neighbor search over embeddings.
    Maintains mapping from keys to embeddings and back.
    """
    def __init__(self, dim: int, use_faiss: bool = True):
        self.dim = dim
        self.use_faiss = use_faiss
        # key list in same order as index vectors
        self.keys: List[str] = []
        # linear store for fallback
        self.embs: List[np.ndarray] = []
        if self.use_faiss:
            # IndexFlatIP for inner-product similarity
            self.index = faiss.IndexFlatIP(dim)

    def add(self, key: str, emb: np.ndarray):
        """Add a new vector to the index."""
        vec = emb.reshape(1, -1).astype(np.float32)
        self.keys.append(key)
        self.embs.append(emb)
        if self.use_faiss:
            self.index.add(vec)

    def remove(self, key: str):
        """Remove a key (and its embedding) from the index. Requires rebuild."""
        # find position
        try:
            idx = self.keys.index(key)
        except ValueError:
            return
        # drop
        self.keys.pop(idx)
        self.embs.pop(idx)
        if self.use_faiss:
            # rebuild index
            self.index.reset()
            all_vecs = np.stack(self.embs).astype(np.float32)
            if len(all_vecs):
                self.index.add(all_vecs)

    def query(self, emb: np.ndarray, topk: int = 1) -> Tuple[Optional[str], float]:
        """
        Search for the most similar cached embedding to `emb`.
        Returns (best_key, best_sim).
        If cache empty, returns (None, -inf).
        """
        if not self.keys:
            return None, -np.inf
        vec = emb.reshape(1, -1).astype(np.float32)
        if self.use_faiss:
            D, I = self.index.search(vec, min(len(self.keys), topk))
            best_sim = float(D[0,0])
            best_key = self.keys[I[0,0]]
            return best_key, best_sim
        else:
            # linear scan
            best_key, best_sim = None, -np.inf
            for k, e in zip(self.keys, self.embs):
                sim = float(np.dot(e, emb))
                if sim > best_sim:
                    best_sim = sim
                    best_key = k
            return best_key, best_sim

class BaseSimilarityCache:
    """
    Base class for similarity caches. You can register observers
    that riceveranno eventi: 'hit', 'miss', 'add', 'evict'.
    """
    def __init__(
        self, capacity: int, threshold: float, dim: int, use_faiss: bool = True
    ):
        self.capacity = capacity
        self.threshold = threshold
        self.index = SimilarityIndex(dim, use_faiss)
        self._observers: List[Callable[[str, str, "BaseSimilarityCache"], None]] = []

    def register_observer(self, fn: Callable[[str, str, 'BaseSimilarityCache'], None]):
        """fn(event_type, key, cache)"""
        self._observers.append(fn)

    def _notify(self, event_type: str, key: Optional[str]):
        for fn in self._observers:
            fn(event_type, key, self)

    def query(self, key: str, emb: np.ndarray) -> bool:
        best_key, best_sim = self.index.query(emb)
        if best_sim >= self.threshold:
            self._on_hit(best_key)
            self._notify('hit', best_key)
            return True
        else:
            self._notify('miss', key)
            self._on_miss(key, emb)
            return False

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
    def _on_hit(self, key: str):        raise NotImplementedError
    def _select_eviction(self) -> str:  raise NotImplementedError
