import numpy as np
from BaseCache import BaseSimilarityCache


class GreedyCache(BaseSimilarityCache):
    """
    Greedy similarity cache. Replaces items only when it reduces the expected cost.
    Requires knowledge of request frequencies λ(x).
    """
    def __init__(self, capacity, threshold, dim, backend, lambda_freq = None, ca_func = lambda x, y: np.linalg.norm(x - y), cr = 1.0, adaptive_thresh=False):
        super().__init__(capacity, threshold, dim, backend, adaptive_thresh)
        self.lambda_freq = lambda_freq  # dict: key -> λ(x)
        self.ca_func = ca_func          # Callable: Ca(x, y)
        self.cr = cr                    # float: cost of a miss

        if not self.lambda_freq:
            # define a default frequency if not provided
            self.lambda_freq = {k: 1.0 for k in self.index.keys}

    def query(self, key: str, emb: np.ndarray) -> bool:
        # Check for exact or approximate hit
        best_key, best_sim = self.index.query(emb)

        if self._should_accept(best_key, best_sim):
            self._on_hit(best_key)
            self._notify('hit', best_key, best_sim)
            return True

        # Try greedy replacement
        current_cost = self._expected_cost()
        best_improvement = 0
        best_victim = None

        for victim in self.index.keys:
            temp_keys = list(self.index.keys)
            temp_keys.remove(victim)
            temp_keys.append(key)

            new_cost = self._expected_cost(temp_keys, new_key=key, emb=emb)
            improvement = current_cost - new_cost

            if improvement > best_improvement:
                best_improvement = improvement
                best_victim = victim

        if best_improvement > 0:
            self.index.remove(best_victim)
            self._notify('evict', best_victim)
            self._on_evict(best_victim)

            self.index.add(key, emb)
            self._notify('add', key)
            self._on_add(key)
            return False  # Miss, but now in cache

        # Serve approximate result without cache update
        self._notify('miss', key, best_sim)
        return False

    def _expected_cost(self, keys=None, new_key=None, emb=None):
        if keys is None:
            keys = list(self.index.keys)

        cost = 0.0
        for x, freq in self.lambda_freq.items():
            # If x is the new key, use the embedding provided
            if new_key is not None and x == new_key:
                emb_x = emb
            else:
                emb_x = self.index.get_embedding(x)

            if emb_x is None:
                continue

            min_ca = min(
                self.ca_func(emb_x, self.index.get_embedding(y)) if y != x else 0
                for y in keys
                if self.index.get_embedding(y) is not None
            )
            cost += freq * min(min_ca, self.cr)
        return cost

    def _on_add(self, key: str): pass
    def _on_evict(self, key: str): pass
    def _on_hit(self, key: str): pass
    def _select_eviction(self) -> str: return ""




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
        new_cost = self._expected_cost(temp_keys, new_key=key, emb=emb)
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

