import uuid, threading, time
import numpy as np
import panel as pn
import holoviews as hv
from holoviews.streams import Pipe
from sympy import im
from BaseCache import CacheSimulator  # Il tuo modulo

pn.extension('holoviews')
hv.extension('bokeh')
hv.opts.defaults(hv.opts.Curve(width=600, height=300, tools=['hover'], show_grid=True))

class CacheDashboard:
    def __init__(self, manager, policies, dim=512, num_requests=2000):
        self.manager = manager
        self.policies = {cls.__name__: cls for cls in policies}
        self.dim = dim
        self.num_requests = num_requests
        self.simulator = CacheSimulator()
        self.simulator.subscribe_events(self._handle_event)

        self.pipe_hit = Pipe(data=([], []))
        self.pipe_quality = Pipe(data=([], []))

        self.history = []
        self.current_idx = 0
        self.active_run_id = None
        self.stop_event = threading.Event()

        self.seq_trace = manager.sample_prompts(num_prompts=num_requests, random_order=False)
        self.rand_trace = manager.sample_prompts(num_prompts=num_requests, random_order=True)

        self._build_ui()

    def _build_ui(self):
        self.policy_select = pn.widgets.Select(name='Policy', options=list(self.policies.keys()))
        self.trace_select = pn.widgets.RadioButtonGroup(name='Trace', options=['Sequential', 'Random'], button_type='primary')
        self.capacity = pn.widgets.IntSlider(name='Capacity', start=50, end=1000, value=200)
        self.threshold = pn.widgets.FloatSlider(name='Threshold', start=0.1, end=1.0, step=0.05, value=0.7)

        self.start_button = pn.widgets.Button(name='Start', button_type='success')
        self.stop_button = pn.widgets.Button(name='Stop', button_type='danger')

        self.status_text = pn.pane.Markdown("## Ready")
        self.hit_rate_text = pn.widgets.StaticText(name="Hit Rate", value="0.0%")
        self.occupancy_text = pn.widgets.StaticText(name="Occupancy", value="0%")

        # Grafici
        self.hit_curve = hv.DynamicMap(self._plot_hit, streams=[self.pipe_hit])
        self.qual_curve = hv.DynamicMap(self._plot_quality, streams=[self.pipe_quality])

        self.start_button.on_click(self._on_start)
        self.stop_button.on_click(self._on_stop)

        self.panel = pn.Column(
            pn.Row(self.policy_select, self.trace_select),
            pn.Row(self.capacity, self.threshold),
            pn.Row(self.start_button, self.stop_button),
            self.status_text,
            pn.Row(self.hit_rate_text, self.occupancy_text),
            "### Hit Rate (%)",
            pn.panel(self.hit_curve),
            "### Quality of Hits (%)",
            pn.panel(self.qual_curve)
        )

    def _on_start(self, *_):
        self.history.clear()
        self.current_idx = 0
        self.stop_event.clear()
        self.active_run_id = str(uuid.uuid4())[:8]
        self.pipe_hit.send(([], []))
        self.pipe_quality.send(([], []))
        threading.Thread(target=self._run_simulation, daemon=True).start()

    def _on_stop(self, *_):
        self.stop_event.set()
        self.status_text.object = "## Stopped"

    def _run_simulation(self):
        policy_name = self.policy_select.value
        policy_class = self.policies[policy_name]
        trace = self.seq_trace if self.trace_select.value == "Sequential" else self.rand_trace

        keys = [k for k, _ in trace]
        embeddings = np.array([e for _, e in trace])
        trace_indices = list(range(len(keys)))

        from inspect import signature
        sig = signature(policy_class.__init__)
        args = {
            k: v for k, v in dict(
                capacity=self.capacity.value,
                threshold=self.threshold.value,
                dim=self.dim,
                backend="faiss_flat"
            ).items() if k in sig.parameters
        }

        self.status_text.object = f"## Running {policy_name}"

        self.simulator.run(
            run_id=self.active_run_id,
            policy_name=policy_name,
            policy_class=policy_class,
            policy_args=args,
            embeddings=embeddings,
            keys=keys,
            trace_indices=trace_indices
        )

        self.status_text.object = f"## Finished {policy_name}"

    def _handle_event(self, run_id, event_type, info):
        if run_id != self.active_run_id or self.stop_event.is_set():
            return

        self.current_idx += 1
        is_hit = event_type == 'hit'
        sim = info.get("sim", None)
        occ = info.get("cache_occupancy", 0)
        capacity = self.capacity.value

        self.occupancy_text.value = f"{occ / capacity * 100:.1f}%"

        self.history.append((self.current_idx, is_hit, sim))

        if self.current_idx % 10 == 0:
            xs = [h[0] for h in self.history]
            hits = [sum(1 for j in self.history[:i] if j[1]) / i * 100 for i in xs]
            sims = [np.mean([j[2] for j in self.history[:i] if j[1] and j[2] is not None])*100 if any(j[1] and j[2] for j in self.history[:i]) else 0 for i in xs]

            hr = hits[-1]
            self.hit_rate_text.value = f"{hr:.1f}%"

            self.pipe_hit.send((xs, hits))
            self.pipe_quality.send((xs, sims))

    def _plot_hit(self, data):
        xs, ys = data
        return hv.Curve((xs, ys), 'Request', 'Hit Rate (%)').opts(ylim=(0, 100), xlim=(1, self.num_requests), color="green")

    def _plot_quality(self, data):
        xs, ys = data
        return hv.Curve((xs, ys), 'Request', 'Avg Similarity (%)').opts(ylim=(0, 100), xlim=(1, self.num_requests), color="orange")

    def show(self):
        return self.panel


if __name__ == "__main__":
    from CachePolicy import LRUCache, LFUCache, RNDLRUCache
    from CacheAware import OSACache, GreedyCache
    from CacheUnAware import QLRUDeltaCCache
    from DuelCache_integrated import DuelCache

    policies = [LRUCache, LFUCache, RNDLRUCache, GreedyCache, OSACache, DuelCache, QLRUDeltaCCache, DuelCache]

    from PromptDatasetManager import PromptDatasetManager
    manager = PromptDatasetManager()
    manager.load_local_metadata(
        path="/Users/tella/Workspace/Inria/Data/normalized_embeddings.parquet",
        max_rows=500 # Limita per testare velocemente
    )
    
    dashboard = CacheDashboard(
        manager=manager,
        policies=policies,
        dim=manager.emb_matrix.shape[1],
        num_requests=2000
    )

    pn.serve(dashboard.show(), show=True)