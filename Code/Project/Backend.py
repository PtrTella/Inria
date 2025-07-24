import faiss
import numpy as np
from annoy import AnnoyIndex
from typing import Protocol, Optional, Tuple, List

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

class FAISSFlatIPIndex(ISimilarityIndex):
    """Inner-product brute-force: ottimo per dataset piccoli/medi."""
    def __init__(self, dim: int):
        self.dim = dim
        self._keys: List[str] = []
        self._embs: List[np.ndarray] = []
        self.index = faiss.IndexFlatIP(dim)

    def add(self, key: str, emb: np.ndarray):
        vec = emb.reshape(1, -1).astype(np.float32)
        self._keys.append(key); self._embs.append(emb)
        self.index.add(vec)

    def remove(self, key: str):
        try:
            idx = self._keys.index(key)
            self._keys.pop(idx); self._embs.pop(idx)
            self.index.reset()
            if self._embs:
                self.index.add(np.stack(self._embs).astype(np.float32))
        except ValueError:
            pass

    def query(self, emb: np.ndarray, topk: int = 1) -> Tuple[Optional[str], float]:
        if not self._keys:
            return None, -np.inf
        vec = emb.reshape(1, -1).astype(np.float32)
        D, I = self.index.search(vec, min(len(self._keys), topk))
        return self._keys[I[0][0]], float(D[0][0])

    @property
    def keys(self):
        return self._keys


class FAISSIVFFlatIndex(ISimilarityIndex):
    """Inverted file + flat quantization: scalabile a grandi dataset."""
    def __init__(self, dim: int, nlist: int = 100):
        self.dim = dim
        self.nlist = nlist
        self._keys: List[str] = []
        self._embs: List[np.ndarray] = []
        # crea l’indice IVF; richiede .train() prima di add()
        quantizer = faiss.IndexFlatIP(dim)
        self.index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        self._trained = False

    def add(self, key: str, emb: np.ndarray):
        self._keys.append(key); self._embs.append(emb)
        if not self._trained and len(self._embs) >= self.nlist * 2:
            data = np.stack(self._embs).astype(np.float32)
            self.index.train(data)
            self._trained = True
        if self._trained:
            self.index.add(emb.reshape(1, -1).astype(np.float32))

    def remove(self, key: str):
        # rimuovi e rebuild IVF da zero
        try:
            idx = self._keys.index(key)
            self._keys.pop(idx); self._embs.pop(idx)
            self.index.reset()
            self._trained = False
            all_data = np.stack(self._embs).astype(np.float32)
            if len(self._embs) >= self.nlist * 2:
                self.index.train(all_data); self._trained = True
                self.index.add(all_data)
        except ValueError:
            pass

    def query(self, emb: np.ndarray, topk: int = 1) -> Tuple[Optional[str], float]:
        if not self._keys or not self._trained:
            return None, -np.inf
        vec = emb.reshape(1, -1).astype(np.float32)
        self.index.nprobe = min(10, self.nlist)  # dimensiona il numero di probe
        D, I = self.index.search(vec, topk)
        return self._keys[I[0][0]], float(D[0][0])

    @property
    def keys(self):
        return self._keys


class FAISSHNSWIndex(ISimilarityIndex):
    """Grafo HNSW: ottimo bilanciamento tra velocità e qualità."""
    def __init__(self, dim: int, M: int = 32, efConstruction: int = 200, efSearch: int = 50):
        self.dim = dim
        self._keys: List[str] = []
        self._embs: List[np.ndarray] = []
        # crea grafo HNSW per IP
        self.index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = efConstruction
        self.index.hnsw.efSearch = efSearch

    def add(self, key: str, emb: np.ndarray):
        vec = emb.reshape(1, -1).astype(np.float32)
        self._keys.append(key); self._embs.append(emb)
        self.index.add(vec)

    def remove(self, key: str):
        try:
            idx = self._keys.index(key)
            self._keys.pop(idx); self._embs.pop(idx)
            # FAISS HNSWFlat non supporta rimozione: rebuild completo
            self.index = faiss.IndexHNSWFlat(self.dim, self.index.hnsw.M, faiss.METRIC_INNER_PRODUCT)
            self.index.hnsw.efConstruction = self.index.hnsw.efConstruction
            self.index.hnsw.efSearch = self.index.hnsw.efSearch
            self.index.add(np.stack(self._embs).astype(np.float32))
        except ValueError:
            pass

    def query(self, emb: np.ndarray, topk: int = 1) -> Tuple[Optional[str], float]:
        if not self._keys:
            return None, -np.inf
        vec = emb.reshape(1, -1).astype(np.float32)
        D, I = self.index.search(vec, topk)
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
    


from typing import Callable

def create_similarity_index(dim: int, backend: str = "faiss_flat") -> ISimilarityIndex:
    backends: dict[str, Callable[..., ISimilarityIndex]] = {
        "faiss_flat": FAISSFlatIPIndex,
        "faiss_ivf": FAISSIVFFlatIndex,
        "faiss_hnsw": FAISSHNSWIndex,
        "annoy": AnnoyIndexWrapper,
        "linear": LinearIndex,
    }
    if backend not in backends:
        raise ValueError(f"Unknown backend: {backend}, available: {list(backends.keys())}")
    # passaggi di parametri specifici per IVF o HNSW se vuoi esporli
    return backends[backend](dim)

