import faiss
import numpy as np
from annoy import AnnoyIndex
from typing import Protocol, Optional, Tuple

class ISimilarityIndex(Protocol):
    def add(self, key: str, emb: np.ndarray): ...
    def remove(self, key: str): ...
    def query(self, emb: np.ndarray, topk: int = 1) -> Tuple[Optional[str], float]: ...
    @property
    def keys(self): ...



class LinearIndex(ISimilarityIndex):
    def __init__(self, dim: int):
        self.dim = dim
        self._keys = []
        self._embs = []

    def add(self, key, emb):
        self._keys.append(key)
        self._embs.append(emb)

    def remove(self, key):
        try:
            idx = self._keys.index(key)
            self._keys.pop(idx)
            self._embs.pop(idx)
        except ValueError:
            pass

    def query(self, emb, topk=1):
        if not self._keys:
            return None, -np.inf
        best_key, best_sim = None, -np.inf
        for k, e in zip(self._keys, self._embs):
            sim = float(np.dot(emb, e))
            if sim > best_sim:
                best_sim = sim
                best_key = k
        return best_key, best_sim

    @property
    def keys(self):
        return self._keys

class FAISSIndex(ISimilarityIndex):
    def __init__(self, dim: int):
        self.dim = dim
        self._keys = []
        self._embs = []
        self.index = faiss.IndexFlatIP(dim)

    def add(self, key, emb):
        vec = emb.reshape(1, -1).astype(np.float32)
        self._keys.append(key)
        self._embs.append(emb)
        self.index.add(vec)

    def remove(self, key):
        try:
            idx = self._keys.index(key)
            self._keys.pop(idx)
            self._embs.pop(idx)
            self.index.reset()
            if self._embs:
                self.index.add(np.stack(self._embs).astype(np.float32))
        except ValueError:
            pass

    def query(self, emb, topk=1):
        if not self._keys:
            return None, -np.inf
        vec = emb.reshape(1, -1).astype(np.float32)
        D, I = self.index.search(vec, min(len(self._keys), topk))
        return self._keys[I[0][0]], float(D[0][0])

    @property
    def keys(self):
        return self._keys

class AnnoyIndexWrapper(ISimilarityIndex):
    def __init__(self, dim: int, n_trees: int = 10):
        self.dim = dim
        self._keys = []
        self._embs = []
        self._index = AnnoyIndex(dim, 'angular')
        self.n_trees = n_trees
        self._built = False

    def add(self, key, emb):
        self._keys.append(key)
        self._embs.append(emb)
        self._built = False  # mark as needing rebuild

    def remove(self, key):
        try:
            idx = self._keys.index(key)
            self._keys.pop(idx)
            self._embs.pop(idx)
            self._built = False
        except ValueError:
            pass

    def _rebuild(self):
        self._index = AnnoyIndex(self.dim, 'angular')
        for i, emb in enumerate(self._embs):
            self._index.add_item(i, emb)
        self._index.build(self.n_trees)
        self._built = True

    def query(self, emb, topk=1):
        if not self._built:
            self._rebuild()
        if not self._keys:
            return None, -np.inf
        idxs = self._index.get_nns_by_vector(emb, topk)
        best_idx = idxs[0]
        key = self._keys[best_idx]
        sim = float(np.dot(self._embs[best_idx], emb))  # cosine similarity
        return key, sim

    @property
    def keys(self):
        return self._keys
    

import scann
import numpy as np
from typing import List, Optional, Tuple

class ScaNNIndex:
    def __init__(self, dim: int):
        self.dim = dim
        self._keys: List[str] = []
        self._embs: List[np.ndarray] = []
        self._searcher = None
        self._built = False

    def add(self, key: str, emb: np.ndarray):
        self._keys.append(key)
        self._embs.append(emb)
        self._built = False

    def remove(self, key: str):
        try:
            idx = self._keys.index(key)
            self._keys.pop(idx)
            self._embs.pop(idx)
            self._built = False
        except ValueError:
            pass

    def _rebuild(self):
        if not self._embs:
            self._searcher = None
            return
        data = np.stack(self._embs).astype(np.float32)
        self._searcher = scann.scann_ops_pybind.builder(data, 10, "dot_product")\
                        .score_ah(2, anisotropic_quantization_threshold=0.2)\
                        .reorder(100).build()
        self._built = True

    def query(self, emb: np.ndarray, topk: int = 1) -> Tuple[Optional[str], float]:
        if not self._built:
            self._rebuild()
        if not self._keys or self._searcher is None:
            return None, -np.inf
        neighbors, distances = self._searcher.search(emb.astype(np.float32))
        idx = neighbors[0]
        key = self._keys[idx]
        sim = float(np.dot(self._embs[idx], emb))
        return key, sim

    @property
    def keys(self):
        return self._keys


def create_similarity_index(dim: int, backend: str = "faiss") -> ISimilarityIndex:
    if backend == "faiss":
        return FAISSIndex(dim)
    elif backend == "annoy":
        return AnnoyIndexWrapper(dim)
    elif backend == "linear":
        return LinearIndex(dim)
    else:
        raise ValueError(f"Unknown backend: {backend}. Supported: 'faiss', 'annoy', 'linear'.")
