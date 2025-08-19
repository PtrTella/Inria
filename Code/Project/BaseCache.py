import time
import numpy as np
import pandas as pd
from Backend_patched import create_similarity_index
from typing import Callable, List, Optional, Type



class BaseSimilarityCache:
    """
    Base class for similarity caches. You can register observers
    that riceveranno eventi: 'hit', 'miss', 'add', 'evict'.
    """
    def __init__(
        self, capacity: int, threshold: float, dim: int, backend: str = "faiss_flat", adaptive_thresh = False
    ):
        self.capacity = capacity
        self.threshold = threshold
        self.adaptive_thresh = adaptive_thresh
        self.index = create_similarity_index(dim, backend)
        self._observers: List[Callable[[str, str, "BaseSimilarityCache"], None]] = []

    def register_observer(self, fn: Callable[[str, str, 'BaseSimilarityCache'], None]):
        """fn(event_type, key, cache)"""
        self._observers.append(fn)

    def _notify(self, event_type: str, key: Optional[str], sim: Optional[float] = None):
        for fn in self._observers:
            fn(event_type, key, self, sim)

    def query(self, key: str, emb: np.ndarray) -> bool:
        best_key, best_sim = self.index.query(emb)

        if self._should_accept(best_key, best_sim):
            self._on_hit(best_key)
            self._notify('hit', best_key, best_sim)
            return True
        else:
            self._notify('miss', key, best_sim)
            self._on_miss(key, emb)
            return False

    def adaptive_acceptance(method):
        def wrapped(self, key, sim):
            if self.adaptive_thresh:
                occ = len(self.index.keys) / self.capacity
                dyn_thresh = min(1.0, self.threshold * (1 + self.adaptive_thresh * occ))
                self.threshold = dyn_thresh
            return method(self, key, sim)
        return wrapped

    @adaptive_acceptance
    def _should_accept(self, key: Optional[str], sim: float) -> bool:
        return sim >= self.threshold  # Default behaviour

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
    def _on_hit(self, key: str):        pass
    def _select_eviction(self) -> str:  pass


class CacheSimulator:
    def __init__(self):
        self.runs = {}        # run_id → metadata
        self.histories = {}   # run_id → list of events (dicts)
        self.current_run = None
        self._gui_subscribers = []

    def run(self, run_id, policy_name, policy_class : Type[BaseSimilarityCache], policy_args, embeddings, keys, trace_indices):

        metadata = {
            'run_id': run_id,
            'policy': policy_name,
            'params': policy_args,
            'start_time': time.time(),
            'hit': 0,
            'miss': 0,
            'total': 0,
        }

        self.runs[run_id] = metadata
        self.histories[run_id] = []

        # Crea la cache e collega l'observer corretto
        def observer(event_type, key, cache_obj, sim):
            self._observe(run_id, event_type, key, cache_obj, sim)

        cache = policy_class(**policy_args)
        cache.register_observer(observer)

        for step, idx in enumerate(trace_indices):
            key = keys[idx]
            emb = embeddings[idx]
            result = cache.query(key, emb)
            metadata['total'] += 1
            if result:
                metadata['hit'] += 1
            else:
                metadata['miss'] += 1

        metadata['end_time'] = time.time()
        metadata['duration'] = metadata['end_time'] - metadata['start_time']
        self.current_run = run_id

    def _observe(self, run_id, event_type, key, cache, sim):
        # Salva l'evento
        entry = dict(
            event=event_type,
            key=key,
            sim=sim,
            timestamp=time.time(),
            cache_occupancy=len(cache.index.keys),
        )
        self.histories[run_id].append(entry)

        # Notifica eventuali dashboard
        for cb in self._gui_subscribers:
            cb(run_id, event_type, entry)

    def get_summary(self, run_id):
        meta = self.runs[run_id]
        events = self.histories[run_id]
        sims = [e['sim'] for e in events if e['sim'] is not None]
        avg_sim = np.mean(sims) if sims else 0.0
        return {
            'run_id': run_id,
            'policy': meta['policy'],
            'hit_rate': meta['hit'] / meta['total'] if meta['total'] > 0 else 0,
            'miss_rate': meta['miss'] / meta['total'] if meta['total'] > 0 else 0,
            'avg_similarity': avg_sim,
            'duration': meta.get('duration', 0.0),
            'params': meta['params'],
        }

    def get_history(self, run_id) -> pd.DataFrame:
        return pd.DataFrame(self.histories[run_id])

    def export(self, path, format='csv'):
        if self.current_run is None:
            raise ValueError("No run to export.")
        df = self.get_history(self.current_run)
        if format == 'csv':
            df.to_csv(path, index=False)
        elif format == 'json':
            df.to_json(path, orient='records')
        elif format == 'parquet':
            df.to_parquet(path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def subscribe_events(self, callback):
        """
        Register a callback to receive cache event updates: 
        fn(run_id, event_type, event_data_dict)
        """
        self._gui_subscribers.append(callback)
