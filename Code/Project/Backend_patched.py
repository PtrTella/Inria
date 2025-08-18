
from __future__ import annotations

from typing import Protocol, Optional, Tuple, List, Dict, Callable
import numpy as np

# Try to import faiss if available
try:
    import faiss  # type: ignore
    _FAISS_AVAILABLE = True
except Exception:  # pragma: no cover
    faiss = None  # type: ignore
    _FAISS_AVAILABLE = False

# Try to import annoy if available
try:
    from annoy import AnnoyIndex  # type: ignore
    _ANNOY_AVAILABLE = True
except Exception:  # pragma: no cover
    AnnoyIndex = None  # type: ignore
    _ANNOY_AVAILABLE = False


def _ensure_f32(vec: np.ndarray, dim: int) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    if arr.ndim != 1 or arr.shape[0] != dim:
        raise ValueError(f"Embedding must be shape ({dim},), got {arr.shape}")
    return arr


def _l2norm(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x)
    if n < eps:
        return x.copy()
    return x / n


class ISimilarityIndex(Protocol):
    def add(self, key: str, emb: np.ndarray) -> None: ...
    def remove(self, key: str) -> None: ...
    def query(self, emb: np.ndarray, topk: int = 1) -> Tuple[Optional[str], float]: ...
    def get_embedding(self, key: str) -> Optional[np.ndarray]: ...
    def query_topk(self, emb: np.ndarray, k: int) -> List[Tuple[str, float]]: ...
    @property
    def keys(self) -> List[str]: ...


class LinearIndex(ISimilarityIndex):
    """Pure-Python/NumPy index. Always available; useful for tests and small data."""
    def __init__(self, dim: int):
        self.dim = dim
        self._store: Dict[str, np.ndarray] = {}  # normalized vectors

    def add(self, key: str, emb: np.ndarray) -> None:
        v = _ensure_f32(emb, self.dim)
        self._store[key] = _l2norm(v)

    def remove(self, key: str) -> None:
        self._store.pop(key, None)

    def query(self, emb: np.ndarray, topk: int = 1) -> Tuple[Optional[str], float]:
        if not self._store:
            return None, -1.0
        q = _l2norm(_ensure_f32(emb, self.dim))
        # cosine similarity = dot of normalized vectors
        best_key = None
        best_sim = -1.0
        for k, v in self._store.items():
            sim = float(np.dot(q, v))
            if sim > best_sim:
                best_sim = sim
                best_key = k
        return best_key, best_sim

    def get_embedding(self, key: str) -> Optional[np.ndarray]:
        """Returns the *normalized* embedding stored for `key` (float32, shape (dim,)), or None."""
        v = self._store.get(key)
        return None if v is None else v.copy()
    
    # LinearIndex.query_topk
    def query_topk(self, emb, k):
        if not self._store: return []
        q = _l2norm(_ensure_f32(emb, self.dim))
        sims = [(key, float(np.dot(q, v))) for key, v in self._store.items()]
        sims.sort(key=lambda kv: kv[1], reverse=True)
        return sims[:max(1, k)]

    @property
    def keys(self) -> List[str]:
        return list(self._store.keys())


class FAISSFlatIPIndex(ISimilarityIndex):
    """
    FAISS Flat + inner product, wrapped with an ID map to support add/remove by key.
    Vectors are L2-normalized so inner product == cosine similarity.
    """
    def __init__(self, dim: int):
        if not _FAISS_AVAILABLE:
            raise RuntimeError("faiss is not available; install faiss-cpu or faiss-gpu, or use backend='linear'.")
        self.dim = dim
        self._index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))  # inner product
        self._key_to_id: Dict[str, int] = {}
        self._id_to_key: Dict[int, str] = {}
        self._key_to_emb: Dict[str, np.ndarray] = {}  # store normalized vectors
        self._next_id: int = 1

    def add(self, key: str, emb: np.ndarray) -> None:
        v = _l2norm(_ensure_f32(emb, self.dim))
        # If key exists, replace
        if key in self._key_to_id:
            self.remove(key)
        id_ = self._next_id
        self._next_id += 1
        ids = np.array([id_], dtype=np.int64)
        vecs = v.reshape(1, -1).astype(np.float32, copy=False)
        self._index.add_with_ids(vecs, ids)
        self._key_to_id[key] = id_
        self._id_to_key[id_] = key
        self._key_to_emb[key] = v

    def remove(self, key: str) -> None:
        id_ = self._key_to_id.pop(key, None)
        if id_ is None:
            return
        try:
            self._index.remove_ids(np.array([id_], dtype=np.int64))
        finally:
            self._id_to_key.pop(id_, None)
            self._key_to_emb.pop(key, None)

    def query(self, emb: np.ndarray, topk: int = 1) -> Tuple[Optional[str], float]:
        if not self._key_to_id:
            return None, -1.0
        q = _l2norm(_ensure_f32(emb, self.dim))
        D, I = self._index.search(q.reshape(1, -1).astype(np.float32, copy=False), max(1, topk))
        if I.size == 0 or I[0, 0] < 0:
            return None, -1.0
        id0 = int(I[0, 0])
        key0 = self._id_to_key.get(id0)
        sim0 = float(D[0, 0]) if key0 is not None else -1.0
        return key0, sim0

    def get_embedding(self, key: str) -> Optional[np.ndarray]:
        v = self._key_to_emb.get(key)
        return None if v is None else v.copy()
    
    # FAISSFlatIPIndex.query_topk
    def query_topk(self, emb, k):
        if not self._key_to_id: return []
        q = _l2norm(_ensure_f32(emb, self.dim))
        D, I = self._index.search(q.reshape(1, -1).astype(np.float32, copy=False), max(1, k))
        out = []
        for id_, sim in zip(I[0].tolist(), D[0].tolist()):
            if id_ == -1: continue
            key = self._id_to_key.get(int(id_))
            if key is not None:
                out.append((key, float(sim)))
        return out

    @property
    def keys(self) -> List[str]:
        return list(self._key_to_id.keys())


class FAISSIVFFlatIndex(FAISSFlatIPIndex):
    """
    Same API, different FAISS index: IVF with inner product.
    You probably want to train this with representative data before use.
    """
    def __init__(self, dim: int, nlist: int = 256):
        if not _FAISS_AVAILABLE:
            raise RuntimeError("faiss is not available; install faiss-cpu or faiss-gpu, or use backend='linear'.")
        self.dim = dim
        quantizer = faiss.IndexFlatIP(dim)
        self._index = faiss.IndexIDMap(faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT))
        self._key_to_id = {}
        self._id_to_key = {}
        self._key_to_emb = {}
        self._next_id = 1
        # Note: caller must call `train` before first add; we hide that detail by training lazily.
        self._trained = False

    def add(self, key: str, emb: np.ndarray) -> None:
        v = _l2norm(_ensure_f32(emb, self.dim))
        if key in self._key_to_id:
            self.remove(key)
        if not self._trained:
            # train on the first 1000 vectors accumulated (including this one) for simplicity
            # In production, expose a proper train API.
            vecs = [v] + [e for e in self._key_to_emb.values()][:999]
            xb = np.stack(vecs, axis=0).astype(np.float32, copy=False)
            self._index.train(xb)
            self._trained = True
        id_ = self._next_id
        self._next_id += 1
        self._index.add_with_ids(v.reshape(1, -1), np.array([id_], dtype=np.int64))
        self._key_to_id[key] = id_
        self._id_to_key[id_] = key
        self._key_to_emb[key] = v


class FAISSHNSWIndex(FAISSFlatIPIndex):
    """FAISS HNSW with inner product; same wrapping."""
    def __init__(self, dim: int, M: int = 32, efSearch: int = 128, efConstruction: int = 200):
        if not _FAISS_AVAILABLE:
            raise RuntimeError("faiss is not available; install faiss-cpu or faiss-gpu, or use backend='linear'.")
        self.dim = dim
        self._index = faiss.IndexIDMap(faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT))
        self._index.hnsw.efSearch = efSearch
        self._index.hnsw.efConstruction = efConstruction
        self._key_to_id = {}
        self._id_to_key = {}
        self._key_to_emb = {}
        self._next_id = 1


class AnnoyIndexWrapper(ISimilarityIndex):
    """
    Annoy wrapper. Annoy doesn't support deletions; we rebuild lazily.
    Stored vectors are normalized so angular distance matches cosine.
    """
    def __init__(self, dim: int, n_trees: int = 10):
        if not _ANNOY_AVAILABLE:
            raise RuntimeError("annoy is not available; pip install annoy, or use another backend.")
        self.dim = dim
        self._n_trees = n_trees
        self._index = AnnoyIndex(dim, metric='angular')
        self._key_to_id: Dict[str, int] = {}
        self._id_to_key: Dict[int, str] = {}
        self._key_to_emb: Dict[str, np.ndarray] = {}
        self._next_id = 1
        self._built = False
        self._dirty = False

    def _rebuild_if_needed(self):
        if not self._dirty:
            return
        self._index = AnnoyIndex(self.dim, metric='angular')
        for k, v in self._key_to_emb.items():
            id_ = self._key_to_id[k]
            self._index.add_item(id_, v.tolist())
        self._index.build(self._n_trees)
        self._built = True
        self._dirty = False

    def add(self, key: str, emb: np.ndarray) -> None:
        v = _l2norm(_ensure_f32(emb, self.dim))
        if key in self._key_to_id:
            self.remove(key)
        id_ = self._next_id
        self._next_id += 1
        self._key_to_id[key] = id_
        self._id_to_key[id_] = key
        self._key_to_emb[key] = v
        self._index.add_item(id_, v.tolist())
        self._built = False
        self._dirty = True

    def remove(self, key: str) -> None:
        id_ = self._key_to_id.pop(key, None)
        if id_ is None:
            return
        self._id_to_key.pop(id_, None)
        self._key_to_emb.pop(key, None)
        self._dirty = True

    def query(self, emb: np.ndarray, topk: int = 1) -> Tuple[Optional[str], float]:
        if not self._key_to_id:
            return None, -1.0
        self._rebuild_if_needed()
        if not self._built:
            self._index.build(self._n_trees)
            self._built = True
        q = _l2norm(_ensure_f32(emb, self.dim))
        ids, dists = self._index.get_nns_by_vector(q.tolist(), max(1, topk), include_distances=True)
        if not ids:
            return None, -1.0
        # Annoy returns angular distance; convert to cosine similarity:
        # cos(theta) = 1 - (dist^2)/2  (approx), but annoy's "angular" distance is 2 * (1 - cos), so cos = 1 - dist/2
        # However Annoy's docs: distance = 2*(1-cosine), so similarity = 1 - distance/2
        sim0 = 1.0 - float(dists[0]) / 2.0
        key0 = self._id_to_key.get(int(ids[0]))
        return key0, sim0

    def get_embedding(self, key: str) -> Optional[np.ndarray]:
        v = self._key_to_emb.get(key)
        return None if v is None else v.copy()

    @property
    def keys(self) -> List[str]:
        return list(self._key_to_id.keys())


def create_similarity_index(dim: int, backend: str = "faiss_flat") -> ISimilarityIndex:
    """
    Factory to create an index. Valid backend values:
      - 'faiss_flat'  (recommended when faiss is installed)
      - 'faiss_ivf'
      - 'faiss_hnsw'
      - 'annoy'
      - 'linear'      (always available)
    """
    backends: Dict[str, Callable[..., ISimilarityIndex]] = {
        "faiss_flat": FAISSFlatIPIndex,
        "faiss_ivf": FAISSIVFFlatIndex,
        "faiss_hnsw": FAISSHNSWIndex,
        "annoy": AnnoyIndexWrapper,
        "linear": LinearIndex,
    }
    if backend not in backends:
        raise ValueError(f"Unknown backend: {backend}, available: {list(backends.keys())}")
    if backend.startswith("faiss") and not _FAISS_AVAILABLE:
        # graceful fallback
        return LinearIndex(dim)
    if backend == "annoy" and not _ANNOY_AVAILABLE:
        return LinearIndex(dim)
    return backends[backend](dim)
