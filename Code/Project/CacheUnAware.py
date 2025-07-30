import random
import numpy as np
from collections import deque

from BaseCache import BaseSimilarityCache

class QLRUDeltaCCache(BaseSimilarityCache):
    """
    qLRU-ΔC similarity cache.
    Probabilistic insertion and refresh based on approximation cost.
    """
    def __init__(self, capacity, threshold, dim, backend, ca_func = lambda x, y: np.linalg.norm(x - y), cr = 1.0, q=0.2, adaptive_thresh=False):
        super().__init__(capacity, threshold, dim, backend, adaptive_thresh)
        self.ca_func = ca_func  # Callable: Ca(x, y)
        self.cr = cr            # retrieval cost
        self.q = q              # admission probability
        self.queue = deque()    # LRU queue

    def query(self, key: str, emb: np.ndarray) -> bool:
        best_key, best_sim = self.index.query(emb)

        if best_key and self._should_accept(best_key, best_sim):
            # Hit approssimato o esatto
            emb_z = self.index.get_embedding(best_key)
            approx_cost = self.ca_func(emb, emb_z)

            # Refresh z con prob. proporzionale al saving
            if random.random() < (self.cr - approx_cost) / self.cr:
                if best_key in self.queue:
                    self.queue.remove(best_key)
                    self.queue.appendleft(best_key)

            # Inserisci x anche se è un hit approssimato, con bassa prob.
            if random.random() < self.q * (approx_cost / self.cr):
                self._insert(key, emb)

            self._on_hit(best_key)
            self._notify('hit', best_key, best_sim)
            return True

        # Miss totale: inserisci x con probabilità q
        if random.random() < self.q:
            self._insert(key, emb)

        self._notify('miss', key, best_sim)
        return False

    def _insert(self, key, emb):
        if key in self.queue:
            self.queue.remove(key)
        self.queue.appendleft(key)
        self.index.add(key, emb)
        self._notify('add', key)
        self._on_add(key)

        # Evict se superiamo la capacità
        if len(self.queue) > self.capacity:
            evicted = self.queue.pop()
            self.index.remove(evicted)
            self._notify('evict', evicted)
            self._on_evict(evicted)

    def _on_add(self, key: str): pass
    def _on_evict(self, key: str): pass
    def _on_hit(self, key: str): pass


from typing import Optional, Tuple, List
import numpy as np

from BaseCache import BaseSimilarityCache

def _ensure_vec(x: np.ndarray, dim: int) -> np.ndarray:
    v = np.asarray(x, dtype=np.float32).reshape(-1)
    if v.shape[0] != dim:
        raise ValueError(f"Embedding must have shape ({dim},), got {v.shape}")
    return v

def _l2norm(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(x))
    if n < eps:
        return x.astype(np.float32, copy=True)
    return (x / n).astype(np.float32, copy=False)


class DuelCache(BaseSimilarityCache):
    """
    Unaware policy-dueling cache.
    Confronta due politiche (A e B) in parallelo su indici-ombra privati e
    serve il traffico usando quella con hit-rate migliore su una finestra scorrevole.
    Le mutazioni reali avvengono sull'indice di DuelCache; per decidere l'ammissione
    riutilizziamo la logica della policy attiva applicata al nostro indice.
    """

    def __init__(
        self,
        capacity: int,
        threshold: float,
        dim: int,
        backend: str,
        policy_a_cls,
        policy_b_cls,
        policy_a_kwargs: Optional[dict] = None,
        policy_b_kwargs: Optional[dict] = None,
        eval_window: int = 200,
        split: Tuple[int, int] = (1, 1),
        adaptive_thresh: bool = False,
    ):
        super().__init__(capacity, threshold, dim, backend, adaptive_thresh)
        self.eval_window = int(eval_window)
        self.split = split
        self._active = 'A'
        self._history: List[Tuple[bool, bool]] = []

        # Crea le due politiche ombra con loro indici
        capA = max(1, capacity * split[0] // max(1, sum(split)))
        capB = max(1, capacity * split[1] // max(1, sum(split)))

        kwargsA = dict(policy_a_kwargs or {})
        kwargsB = dict(policy_b_kwargs or {})

        self._shadowA = policy_a_cls(capA, threshold, dim, backend, **kwargsA)
        self._shadowB = policy_b_cls(capB, threshold, dim, backend, **kwargsB)

    def _switch_if_needed(self):
        if len(self._history) < self.eval_window:
            return
        window = self._history[-self.eval_window:]
        hitA = sum(1 for a, _ in window if a)
        hitB = sum(1 for _, b in window if b)
        if hitA > hitB and self._active != 'A':
            self._active = 'A'
            self._notify('policy', 'activate_A')
        elif hitB > hitA and self._active != 'B':
            self._active = 'B'
            self._notify('policy', 'activate_B')

    def query(self, key: str, emb: np.ndarray) -> bool:
        emb = _l2norm(_ensure_vec(emb, self.dim))

        # Valuta su ombra
        hitA = self._shadowA.query(key, emb)
        hitB = self._shadowB.query(key, emb)
        self._history.append((hitA, hitB))
        self._switch_if_needed()

        # Serve tramite l'indice reale usando la logica della policy attiva
        k0, s0 = self.index.query(emb, topk=1)
        if k0 is not None and s0 >= self.threshold:
            self._notify('hit', k0, s0)
            self._on_hit(k0)
            return True

        # Miss: applichiamo la logica della policy attiva al nostro indice
        active = self._shadowA if self._active == 'A' else self._shadowB

        # Esegue il .query della classe attiva, ma passando self (DuelCache) come contesto
        # in modo che le add/evict avvengano sul nostro indice.
        return active.__class__.query(self, key, emb)

    # hook opzionali
    def _on_add(self, key: str): pass
    def _on_evict(self, key: str): pass
    def _on_hit(self, key: str): pass
