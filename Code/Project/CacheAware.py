import numpy as np
from BaseCache import BaseSimilarityCache

def _ensure_unit(x: np.ndarray, eps=1e-12):
    x = x.astype(np.float32, copy=False).reshape(-1)
    n = float(np.linalg.norm(x))
    if n <= eps: return x
    if abs(n - 1.0) <= 5e-3: return x
    return (x / n).astype(np.float32, copy=False)

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))

def _ca_from_sim(sim: float) -> float:
    # Ca in [0,1]
    return max(0.0, min(1.0, (1.0 - float(sim))*0.5))

class GreedyCache(BaseSimilarityCache):
    """
    Greedy λ-aware (Neglia et al.).
    - lambda_freq: dict key->peso (normalizzato o no; qui non è necessario normalizzarla a 1)
    - ca_func: se None usa (1 - cos)/2 coerente con threshold/FAISS
    - cr: costo miss nella stessa scala di Ca (default 1.0)
    - embedding_lookup: funzione per ottenere l'embedding di x anche se x non è in cache
    """
    def __init__(self, capacity, threshold, dim, backend,
                 lambda_freq=None, ca_func=None, cr=1.0,
                 adaptive_thresh=False, embedding_lookup=None, sample_size=256, rng_seed=None):
        super().__init__(capacity, threshold, dim, backend, adaptive_thresh)
        self.lambda_freq = dict(lambda_freq or {})  # NON derivarla dalle chiavi in cache
        self.ca_func = (lambda x,y: _ca_from_sim(_cosine(x,y))) if ca_func is None else ca_func
        self.cr = float(cr)
        self.embedding_lookup = embedding_lookup   # <- nuova
        self.sample_size = sample_size
        self._rng = np.random.default_rng(rng_seed)

    def query(self, key: str, emb: np.ndarray) -> bool:
        emb = _ensure_unit(emb)

        best_key, best_sim = self.index.query(emb)
        if self._should_accept(best_key, best_sim):
            self._on_hit(best_key)
            self._notify('hit', best_key, best_sim)
            return True

        # Miss: se c'è spazio, ammetti subito (riempimento iniziale)
        if len(self.index.keys) < self.capacity:
            self.index.add(key, emb)
            self._notify('add', key, None)
            self._on_add(key)
            self._notify('miss', key, best_sim)
            return False

        # Costo attuale
        current_cost = self._expected_cost()

        best_improvement = 0.0
        best_victim = None

        cache_keys = list(self.index.keys)
        for victim in cache_keys:
            tmp = [k for k in cache_keys if k != victim] + [key]
            new_cost = self._expected_cost(keys=tmp, injected=(key, emb))
            improvement = current_cost - new_cost
            if improvement > best_improvement:
                best_improvement = improvement
                best_victim = victim

        if best_improvement > 1e-12 and best_victim is not None:
            self.index.remove(best_victim)
            self._notify('evict', best_victim, None)
            self._on_evict(best_victim)
            self.index.add(key, emb)
            self._notify('add', key, None)
            self._on_add(key)

        self._notify('miss', key, best_sim)
        return False

    def _expected_cost(self, keys=None, injected=None):
        """
        E[costo(x; keys)] ~ somma_{x in λ} w_x * min(Cr, min_{y in keys: sim>=th} Ca(x,y))
        - injected: (key, emb) per usare un embedding non ancora nell'indice
        """
        if not self.lambda_freq:
            return 0.0  # se non hai λ non puoi fare Greedy → risultato neutro

        if keys is None:
            keys = list(self.index.keys)

        # materializza gli embedding dei y in cache una sola volta
        vecs = {}
        for y in keys:
            if injected is not None and y == injected[0]:
                vecs[y] = injected[1]
            else:
                vy = self.index.get_embedding(y)
                if vy is not None:
                    vecs[y] = vy
        if not vecs:
            return self.cr

        items = list(self.lambda_freq.items())
        if self.sample_size is not None and self.sample_size < len(items):
            lam_keys = np.array([k for k,_ in items], dtype=object)
            lam_w    = np.array([w for _,w in items], dtype=np.float64)
            lam_w    = lam_w/lam_w.sum()
            idx = self._rng.choice(len(lam_keys), size=self.sample_size, replace=False, p=lam_w)
            items = [(lam_keys[i], float(lam_w[i])) for i in idx]

        total = 0.0
        wsum  = 0.0
        for x, w in items:
            ex = self._lookup_embedding(x)
            if ex is None:
                continue
            best_ca = self.cr
            for y, ey in vecs.items():
                sim = _cosine(ex, ey)
                if sim >= self.threshold:  # <<< coerente con la regola di servizio
                    c = self.ca_func(ex, ey)
                    if c < best_ca:
                        best_ca = c
            total += w * min(self.cr, best_ca)
            wsum  += w

        return (total/wsum) if wsum > 0 else self.cr

    def _lookup_embedding(self, key: str):
        if self.embedding_lookup is not None:
            v = self.embedding_lookup(key)
            if v is not None:
                return _ensure_unit(v)
        v = self.index.get_embedding(key)  # solo se è in cache
        return _ensure_unit(v) if v is not None else None

    def _on_add(self, key: str): pass
    def _on_evict(self, key: str): pass
    def _on_hit(self, key: str): pass





import math
import random

class OSACache(GreedyCache):
    """
    Online Simulated Annealing cache. Uses randomness to avoid local minima.
    """
    def __init__(self, capacity, threshold, dim, backend, lambda_freq = None, ca_func = lambda x, y: np.linalg.norm(x - y), cr = 1.0, adaptive_thresh=False):
        super().__init__(capacity, threshold, dim, backend, lambda_freq, ca_func, cr, adaptive_thresh)
        self.t = 1  # request counter
        self.ΔCmax = cr  # can be refined

        if not self.lambda_freq:
            # define a default frequency if not provided
            self.lambda_freq = {k: 1.0 for k in self.index.keys}

    def query(self, key: str, emb: np.ndarray) -> bool:
        self.t += 1
        best_key, best_sim = self.index.query(emb)

        if self._should_accept(best_key, best_sim):
            self._on_hit(best_key)
            self._notify('hit', best_key, best_sim)
            return True

        # Prova una sostituzione casuale
        if not self.index.keys:
            self.index.add(key, emb)
            self._notify('add', key)
            self._on_add(key)
            return False

        victim = random.choice(list(self.index.keys))
        temp_keys = list(self.index.keys)
        temp_keys.remove(victim)
        temp_keys.append(key)

        current_cost = self._expected_cost()
        new_cost = self._expected_cost(keys=temp_keys, injected=(key, emb))
        ΔC = current_cost - new_cost

        T = self.ΔCmax * len(self.index.keys) / (1 + math.log(self.t + 1))
        prob_accept = min(1, math.exp(ΔC / T)) if T > 0 else 0

        if random.random() < prob_accept:
            self.index.remove(victim)
            self._notify('evict', victim)
            self._on_evict(victim)

            self.index.add(key, emb)
            self._notify('add', key)
            self._on_add(key)
            return False

        # Non accettata, rispondi con oggetto più simile
        self._notify('miss', key, best_sim)
        return False

