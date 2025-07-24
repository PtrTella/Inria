import panel as pn
import holoviews as hv
import inspect
from holoviews.streams import Pipe
import numpy as np
import threading, time
from holoviews import opts
from typing import Optional, Type, List
from BaseCache import BaseSimilarityCache
from PromptDatasetManager import PromptDatasetManager

pn.extension('holoviews')
hv.extension('bokeh')

hv.opts.defaults(opts.Curve(width=800, height=400, show_grid=True, tools=[]))

def get_dashboard(manager: PromptDatasetManager, caching_class: List[Type[BaseSimilarityCache]], num_requests: int = 5000, dim: int = 512):

    seq_requests = manager.sample_prompts(num_prompts=num_requests, random_order=False)
    rand_requests = manager.sample_prompts(num_prompts=num_requests, random_order=True)
    trace_len = num_requests

    caching_dict = {cls.__name__: cls for cls in caching_class}

    # Variabili di stato locali
    history = []
    current_idx = 0
    stop_event = threading.Event()
    cache = None
    requests = []
    start_time = None

    # --- Widgets ---
    policy_select   = pn.widgets.Select(name='Policy', options=list(caching_dict.keys()), value=list(caching_dict.keys())[0])
    trace_select    = pn.widgets.RadioButtonGroup(name='Trace', options=['Sequential','Random'], button_type='primary')
    backend_select = pn.widgets.RadioButtonGroup(name='Backend', options=['faiss_flat', 'linear'], button_type='primary')
    capacity_slider = pn.widgets.IntSlider(name='Capacity', start=50, end=10000, step=150, value=500)
    threshold_slider= pn.widgets.FloatSlider(name='Threshold', start=0.5, end=1.0, step=0.05, value=0.8)
    ttl_slider      = pn.widgets.IntSlider(name='TTL', start=10, end=500, step=10, value=100)
    adaptive_thresh_slider = pn.widgets.FloatSlider(name='Adaptive Threshold Coeff.', start=0.0, end=1.0, step=0.2, value=0.0)
    start_button    = pn.widgets.Button(name='Start Live Simulation', button_type='success')
    stop_button     = pn.widgets.Button(name='Stop & Reset', button_type='danger')

    # --- Text widgets for stats ---
    text_hit_rate   = pn.widgets.StaticText(name='Hit Rate', value='0.0%')
    text_slide_rate = pn.widgets.StaticText(name='Sliding Hit Rate (1k)', value='0.0%')
    text_occupancy  = pn.widgets.StaticText(name='Cache Occupancy', value='0/0 (0.0%)')
    text_rps        = pn.widgets.StaticText(name='Execution Time', value='0.0')
    text_hit_quality = pn.widgets.StaticText(name='Hit Quality', value='0.0')



    stream = Pipe(data=[])

    def update_plot(data):
        if not data:
            empty_x, empty_y = [], []
            curve_hr = hv.Curve((empty_x, empty_y), 'Request', 'Hit Rate (%)', label='Hit Rate (%)')
            curve_occ = hv.Curve((empty_x, empty_y), 'Request', 'Occupancy (%)', label='Occupancy (%)')
            return (curve_hr * curve_occ).opts(
                width=800, height=400, show_grid=True,
                ylim=(0, 100), xlim=(1, trace_len),
                legend_position='top_right',
                tools=[]
            )

        xs, hrs_percent, occs_percent = zip(*data)

        curve_hr = hv.Curve((xs, hrs_percent), 'Request', 'Hit Rate (%)', label='Hit Rate (%)').opts(line_width=2)
        curve_occ = hv.Curve((xs, occs_percent), 'Request', 'Occupancy (%)', label='Occupancy (%)').opts(line_width=2, line_dash='dashed')

        return (curve_hr * curve_occ).opts(
            width=800, height=400, show_grid=True,
            ylim=(0, 100), xlim=(1, max(xs)),
            xlabel='Request',
            legend_position='top_right',
        )

    dmap = hv.DynamicMap(update_plot, streams=[stream])

    def push_to_pipe(event_type, key, c, idx, sim: Optional[float] = None):
        is_hit = event_type == 'hit'
        hits = (history[-1][1] * (idx - 1) if history else 0) + (1 if is_hit else 0)
        hr = hits / idx if idx > 0 else 0.0
        occ = len(c.index.keys) / c.capacity

        # Aggiungiamo sim (solo per hit) come quinto valore
        history.append((idx, hr, occ, is_hit, sim if is_hit else None))

        # Sliding HR
        window = history[-1000:] if len(history) > 1000 else history
        sliding_hits = sum(1 for h in window if h[3])
        sliding_hr = sliding_hits / len(window) if window else 0.0

        # Hit similarity media
        sim_vals = [s for (_, _, _, hit, s) in history if hit and s is not None]
        avg_sim = min(sum(sim_vals) / len(sim_vals) if sim_vals else 0.0, 1.0)

        # UI update
        text_hit_rate.value = f"{hr * 100:.1f}%"
        text_slide_rate.value = f"{sliding_hr * 100:.1f}%"
        text_occupancy.value = f"{occ * 100:.1f}%"
        text_hit_quality.value = f"{avg_sim:.4f}"


        # RPS
        elapsed = time.time() - start_time
        rps = idx / elapsed if elapsed > 0 else 0.0
        text_rps.value = f"{elapsed:.1f}"

        if idx % 100 == 0 or idx == len(requests):
            stream.send([(i, h * 100, o * 100) for i, h, o, _, _ in history])


    def create_cache(policy_class, params):
        # Passa i parametri come kwargs, ignora quelli non accettati
        sig = inspect.signature(policy_class.__init__)
        valid_keys = set(sig.parameters.keys()) - {'self'}
        filtered = {k: v for k, v in params.items() if k in valid_keys}
        try:
            return policy_class(**filtered)
        except TypeError:
            # Se il costruttore accetta solo **kwargs
            return policy_class(**params)

    def setup_cache():
        nonlocal cache, requests
        cap = capacity_slider.value
        th = threshold_slider.value
        ttl = ttl_slider.value
        adaptive_thresh = adaptive_thresh_slider.value if adaptive_thresh_slider.value > 0 else False
        trace = seq_requests if trace_select.value=='Sequential' else rand_requests
        backend = backend_select.value
        requests = trace
        policy = policy_select.value

        cache = create_cache(caching_dict[policy], {
            'capacity': cap,
            'threshold': th,
            'dim': dim,
            'adaptive_thresh': adaptive_thresh,
            'backend': backend,
            'ttl': ttl
        })

        def observer(event_type, key, c, sim=None):
            push_to_pipe(event_type, key, c, current_idx, sim)


        cache.register_observer(observer)

    def run_live():
        nonlocal history, start_time, current_idx
        history.clear()
        cache.index = cache.index
        start_time = time.time()
        stop_event.clear()
        for idx, (key, emb) in enumerate(requests, start=1):
            if stop_event.is_set():
                break
            current_idx = idx
            cache.query(key, emb)

    def on_start(event):
        setup_cache()
        stream.send([(0, 0, 0)])
        threading.Thread(target=run_live, daemon=True).start()

    def on_stop(event):
        nonlocal current_idx, history
        stop_event.set()
        history.clear()
        current_idx = 0
        text_hit_rate.value = '0.0%'
        text_slide_rate.value = '0.0%'
        text_occupancy.value = '0.0%'
        text_rps.value = '0.0'
        stream.send([(0, 0, 0)])
        if cache:
            cache.index = cache.index

    start_button.on_click(on_start)
    stop_button.on_click(on_stop)

    dashboard = pn.Column(
        pn.Row(policy_select),
        pn.Row(start_button, stop_button, trace_select, backend_select),
        pn.Row(
            pn.Column(pn.panel(dmap)),
            pn.Column(
                capacity_slider, threshold_slider, adaptive_thresh_slider, ttl_slider, text_hit_rate, text_hit_quality, text_rps, text_occupancy
            ),
        ),
    )

    return dashboard

if __name__ == "__main__":
    # Esempio di utilizzo:
    from PromptDatasetManager import PromptDatasetManager
    manager = PromptDatasetManager()
    manager.load_local_metadata(
        path="/Users/tella/Workspace/Inria/Data/normalized_embeddings.parquet",
        max_rows=100  # Limita per testare velocemente
    )
    from CachePolicy import LRUCache, LFUCache, TTLCache, RNDLRUCache, RNDTTLCache
    dim = manager.emb_matrix.shape[1]
    dashboard = get_dashboard(manager, [LRUCache, LFUCache, TTLCache, RNDLRUCache, RNDTTLCache], num_requests=5000, dim=dim)
    dashboard.show()
