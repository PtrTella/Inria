import panel as pn
import holoviews as hv
from holoviews.streams import Pipe
import numpy as np
import threading
import time
from holoviews import opts

from CachePolicy import LRUCache, LFUCache, TTLCache, RNDLRUCache, RNDTTLCache

pn.extension('holoviews')
hv.extension('bokeh')


class CacheDashboard:
    def __init__(self, cache_factory, data_manager, show=False):
        self.cache_factory = cache_factory  # Callable returning a BaseCache instance
        self.data_manager = data_manager
        self.cache = None
        self.history = []
        self.current_idx = 0
        self.stop_event = threading.Event()
        self.requests = []
        self.start_time = None

        # UI elements
        self.policy_select    = pn.widgets.Select(name='Policy', options=[], value=None)
        self.capacity_slider  = pn.widgets.IntSlider(name='Capacity', start=50, end=2000, step=50, value=500)
        self.threshold_slider = pn.widgets.FloatSlider(name='Threshold', start=0.5, end=1.0, step=0.05, value=0.8)
        self.ttl_slider       = pn.widgets.IntSlider(name='TTL', start=10, end=500, step=10, value=100)
        self.trace_select     = pn.widgets.RadioButtonGroup(name='Trace', options=['Sequential','Random'], button_type='primary')
        self.start_button     = pn.widgets.Button(name='Start Live Simulation', button_type='success')
        self.stop_button      = pn.widgets.Button(name='Stop', button_type='danger')

        self.pipe = Pipe(data=[])
        self.curve = hv.Curve([]).opts(title="Cache Hit Rate", xlabel="Time", ylabel="Hit Rate", height=300, width=600)
        self.dmap = hv.DynamicMap(self.curve, streams=[self.pipe])

        self.start_button.on_click(self._start_simulation)
        self.stop_button.on_click(self._stop_simulation)

        self.layout = pn.Column(
            pn.Row(self.policy_select, self.capacity_slider, self.threshold_slider, self.ttl_slider),
            pn.Row(self.trace_select, self.start_button, self.stop_button),
            self.dmap
        )

        if show:
            self.show()

    def register_policies(self, policy_names):
        self.policy_select.options = policy_names
        if policy_names:
            self.policy_select.value = policy_names[0]

    def _start_simulation(self, *_):
        self.stop_event.clear()
        self.cache = self.cache_factory({
            'policy': self.policy_select.value,
            'capacity': self.capacity_slider.value,
            'threshold': self.threshold_slider.value,
            'ttl': self.ttl_slider.value
        })

        self.requests = self.data_manager.generate(self.trace_select.value)
        self.history = []
        self.current_idx = 0
        self.start_time = time.time()

        threading.Thread(target=self._update_loop, daemon=True).start()

    def _stop_simulation(self, *_):
        self.stop_event.set()

    def _update_loop(self):
        hits = 0
        for i, req in enumerate(self.requests):
            if self.stop_event.is_set():
                break
            if self.cache.get(req) is None:
                self.cache.put(req, req)
            else:
                hits += 1
            self.history.append(hits / (i + 1))
            self.pipe.send(np.column_stack((np.arange(len(self.history)), self.history)))
            time.sleep(0.1)

    def show(self):
        return self.layout.servable()


if __name__ == "__main__":
    
    def my_cache_factory(params):
        policy_cls = {
            'LRU': LRUCache,
            'LFU': LFUCache,
            'TTL': TTLCache,
            'RNDLRU': RNDLRUCache,
            'RNDTTL': RNDTTLCache
        }.get(params['policy'], LRUCache)
        return policy_cls(**params)
    
    from PromptDatasetManager import PromptDatasetManager
    data_manager = PromptDatasetManager()
    data_manager.load_local_metadata(
        path="/Users/tella/Workspace/Inria/Data/normalized_embeddings.parquet",
        max_rows=50  # Limit for quick testing
    )

    dashboard = CacheDashboard(cache_factory=my_cache_factory, data_manager=data_manager)
    dashboard.register_policies(list({
        'LRU', 'LFU', 'TTL', 'RNDLRU', 'RNDTTL'
    })) 
    dashboard.show()