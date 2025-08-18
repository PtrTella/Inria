import random
import numpy as np
from collections import deque

from collections import deque
import random
import numpy as np
from BaseCache import BaseSimilarityCache

def _ensure_unit(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(x, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    if n <= eps: return v
    if abs(n - 1.0) <= 5e-3: return v
    return (v / n).astype(np.float32, copy=False)

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))

def _ca_from_sim(sim: float) -> float:
    # Ca in [0,1] se i vettori sono normalizzati
    return max(0.0, min(1.0, (1.0 - float(sim)) * 0.5))

class QLRUDeltaCCache(BaseSimilarityCache):
    """
    qLRU-ΔC (Neglia et al.).
    - Coda LRU (deque) per l'ordine.
    - Miss: inserimento di x con prob. q.
    - Approx-hit: serve z=argmin Ca(x,y); refresh z con prob. (C(x,S\{z})-Ca(x,z))/Cr;
                  con prob. q*Ca(x,z)/Cr, recupera comunque x e lo inserisce (head).
    """
    def __init__(self, capacity, threshold, dim, backend,
                 ca_func=None, cr=0.2, q=0.2, adaptive_thresh=False,
                 mode='paper'):
        super().__init__(capacity, threshold, dim, backend, adaptive_thresh)
        # default: Ca=(1-cos)/2 coerente con indice coseno
        self.ca = (lambda a,b: _ca_from_sim(_cosine(a,b))) if ca_func is None else ca_func
        self.cr = float(cr)
        self.q = float(q)
        self.queue = deque()     # LRU order: left = MRU, right = LRU
        self.mode = mode         # 'paper' usa Ca<=Cr; 'threshold' usa self.threshold

        # Sanity: per refresh serve 2-NN
        self._has_topk = hasattr(self.index, "query_topk")

    # --- helper per decisione approx-hit coerente
    def _admissible(self, best_sim: float, best_ca: float) -> bool:
        if self.mode == 'paper':
            return best_ca <= self.cr
        return (best_sim is not None) and (best_sim >= float(self.threshold))

    def query(self, key: str, emb: np.ndarray) -> bool:
        x = _ensure_unit(emb)

        # 1-NN (e idealmente 2-NN) per hit e refresh prob
        if self._has_topk:
            topk = self.index.query_topk(x, 2)  # [(key, sim), ...]
        else:
            ## NON SERVE IN TEORIA GESTITO GIA DAL BACKEND
            # fallback: scan lineare
            sims = []
            for k in self.index.keys:
                vk = self.index.get_embedding(k)
                if vk is not None:
                    sims.append((k, _cosine(x, vk)))
            sims.sort(key=lambda kv: kv[1], reverse=True)
            topk = sims[:2]

        hit = False
        best_key, best_sim = (None, None)
        if topk:
            best_key, best_sim = topk[0]
            best_ca = self.ca(x, self.index.get_embedding(best_key))
            if self._admissible(best_sim, best_ca):
                # ---- approx-hit path ----
                z = best_key
                ca_xz = best_ca

                # costo senza z: usa 2° vicino se esiste, altrimenti Cr
                if len(topk) >= 2:
                    _, sim2 = topk[1]
                    ca_alt = self.ca(x, self.index.get_embedding(topk[1][0]))
                    c_wo_z = min(self.cr, ca_alt)
                else:
                    c_wo_z = self.cr

                # refresh prob = (C(x,S\{z}) - Ca(x,z)) / Cr  (clamp [0,1])
                p_refresh = max(0.0, min(1.0, (c_wo_z - ca_xz) / self.cr))
                if random.random() < p_refresh:
                    # move-to-front
                    if z in self.queue:
                        self.queue.remove(z)
                    self.queue.appendleft(z)

                # con prob q*Ca/Cr recupero comunque x e lo inserisco (paper)
                p_insert_x = max(0.0, min(1.0, self.q * (ca_xz / self.cr)))
                if random.random() < p_insert_x:
                    self._insert(key, x)

                self._on_hit(z)
                self._notify('hit', z, best_sim)
                hit = True

        if hit:
            return True

        # ---- miss path (Ca > Cr): inserisci con prob. q ----
        if random.random() < self.q:
            self._insert(key, x)

        self._notify('miss', key, best_sim if best_sim is not None else None)
        return False

    def _insert(self, key: str, v: np.ndarray):
        # LRU insert at head
        if key in self.queue:
            self.queue.remove(key)
        self.queue.appendleft(key)

        self.index.add(key, v)
        self._notify('add', key, None)
        self._on_add(key)

        # Evict se oltre capacità
        if len(self.queue) > self.capacity:
            victim = self.queue.pop()
            self.index.remove(victim)
            self._notify('evict', victim, None)
            self._on_evict(victim)

    # Hooks (observer)
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
