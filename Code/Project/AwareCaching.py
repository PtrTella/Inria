from BaseCache import BaseSimilarityCache
from collections import defaultdict
from typing import Callable, Optional, List
import numpy as np

class AwareSimilarityCache(BaseSimilarityCache):
    def __init__(self, capacity, topk=10):
        super().__init__(capacity)
        self.topk_counter = defaultdict(int)
        self.sim_accumulator = defaultdict(float)
        self.topk = topk

    def observe_topk_result(self, indices, distances):
        for idx, dist in zip(indices[0], distances[0]):
            key = self.index_to_key.get(idx)
            if key:
                self.topk_counter[key] += 1
                self.sim_accumulator[key] += 1 - dist

    def insert(self, key, emb, metadata=None):
        if self.size() >= self.capacity:
            self._evict_lowest_utility()
        self.cache[key] = (emb, metadata)
        self.faiss_index.add(np.expand_dims(emb, axis=0))
        self.index_to_key[self.faiss_index.ntotal - 1] = key

    def _evict_lowest_utility(self):
        lowest = min(self.cache.keys(), key=self.get_utility)
        self.evict(lowest)

    def get_utility(self, key):
        if self.topk_counter[key] == 0:
            return 0.0
        return self.sim_accumulator[key] / self.topk_counter[key]
