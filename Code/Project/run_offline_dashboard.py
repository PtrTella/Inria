#!/usr/bin/env python3
"""
run_offline_dashboard.py

Script locale che sostituisce la Dashboard: esegue sweep di politiche di caching
usando i moduli esistenti (PromptDatasetManager, CacheSimulator, CachePolicy),
raccoglie i dati e salva grafici e csv pronti per il report.

Esempi:
  # Esegue la griglia di default e salva tutto in ./outputs
  python run_offline_dashboard.py --dataset /path/al/diffusiondb_clip.parquet --outdir ./outputs --all

  # Esegue solo la prima riga della griglia (dry-run veloce)
  python run_offline_dashboard.py --dataset /path/al/diffusiondb_clip.parquet --outdir ./outputs
"""
import os, json, argparse, uuid, sys, pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import dei tuoi moduli caricati nella stessa macchina
sys.path.append("/mnt/data")
from PromptDatasetManager import PromptDatasetManager
from BaseCache import CacheSimulator
from CachePolicy import *  # costruiamo un registry dinamico

def build_policy_registry():
    reg = {}
    for name, obj in dict(globals()).items():
        try:
            if name.endswith("Cache") and callable(obj):
                reg[name.replace("Cache","")] = obj
        except Exception:
            pass
    # alias comuni
    for alias in ["LRU","LFU","TTL","TwoLRU","RNDLRU","RNDTTL"]:
        if alias not in reg and alias+"Cache" in reg:
            reg[alias] = reg[alias+"Cache"]
    return reg

def default_runs_grid(dataset_path):
    """Piccola griglia di esempio: personalizzala o passa --runs-jsonl"""
    grid = []
    for policy in ["LRU","LFU","TTL","TwoLRU","RNDLRU","RNDTTL"]:
        for capacity in [1000, 5000, 20000]:
            for thr in [0.70, 0.80, 0.90]:
                row = {
                    "dataset_path": dataset_path,
                    "policy": policy,
                    "capacity": capacity,
                    "threshold": thr,
                    "num_requests": 8000,
                    "trace_mode": "sessions",
                    "session_gap_min": 30,
                    "backend": "faiss_flat",
                    "adaptive_thresh": 0.0
                }
                if policy in ["TTL","RNDTTL"]:
                    row["ttl"] = 256
                if policy == "TwoLRU":
                    row["filter_ratio"] = 0.1
                if policy in ["RNDLRU","RNDTTL"]:
                    row["min_threshold"] = 0.5
                    row["max_threshold"] = 1.0
                grid.append(row)
    return grid


def to_index_array(obj, prompt_to_idx):
    import numpy as _np
    # Normalize to 1D numpy array
    arr = _np.asarray(obj, dtype=object).ravel()
    # If already numeric-ish, try to cast
    try:
        return arr.astype(_np.int64)
    except Exception:
        pass
    # Otherwise assume it's a list of prompt strings -> map to indices
    out = _np.empty(arr.shape[0], dtype=_np.int64)
    for k, v in enumerate(arr):
        try:
            out[k] = prompt_to_idx[v]
        except KeyError:
            raise KeyError(f"Prompt not found in dataset mapping: {v!r}")
    return out

def ensure_index_array(x):
    import numpy as _np
    # sample_prompts / sample_sessions might return (indices, payload) or nested lists
    if isinstance(x, tuple):
        # If first elem looks like indices use it
        if len(x) > 0 and (hasattr(x[0], "__len__") or hasattr(x[0], "shape")):
            x = x[0]
    x = _np.asarray(x)
    if x.ndim > 1:
        x = x.ravel()
    # cast to int indices
    try:
        x = x.astype(_np.int64, copy=False)
    except Exception:
        # fallback: attempt to convert elementwise
        x = _np.array([int(v) for v in x], dtype=_np.int64)
    return x


def make_trace(manager, cfg, prompt_to_idx):
    mode = cfg.get("trace_mode", "sequential")
    n = int(cfg.get("num_requests", 2000))

    def _pairs_to_idx(pairs):
        # pairs: list of (prompt, emb)
        out = []
        for item in pairs:
            if isinstance(item, tuple) and len(item) >= 1:
                p = item[0]
            else:
                # if item is just a prompt string
                p = item
            try:
                out.append(prompt_to_idx[p])
            except KeyError:
                # if p not found, skip
                continue
        return np.asarray(out, dtype=np.int64)

    if mode == "sequential":
        pairs = manager.sample_prompts(num_prompts=n, random_order=False)
        idx = _pairs_to_idx(pairs)
    elif mode == "random":
        pairs = manager.sample_prompts(num_prompts=n, random_order=True)
        idx = _pairs_to_idx(pairs)
    elif mode == "sessions":
        gap = float(cfg.get("session_gap_min", 30))
        maxp = cfg.get("session_max_prompts")
        sess = manager.sample_sessions(gap=gap, num_sessions=None, max_prompts=maxp, random_order=False)
        chunks = []
        for (prompts_chunk, _embs_chunk) in sess:
            # prompts_chunk è un array di stringhe
            for p in prompts_chunk:
                if p in prompt_to_idx:
                    chunks.append(prompt_to_idx[p])
        idx = np.asarray(chunks[:n], dtype=np.int64) if len(chunks) else np.array([], dtype=np.int64)
    else:
        raise ValueError(f"Unsupported trace_mode: {mode}")

    # safety: unique? preserve order
    return ensure_index_array(idx)


def build_runtime_args(cfg, dim):
    args = dict(
        capacity=int(cfg.get("capacity", 10_000)),
        threshold=float(cfg.get("threshold", 0.80)),
        dim=int(dim),
        backend=str(cfg.get("backend", "faiss_flat")),
        adaptive_thresh=float(cfg.get("adaptive_thresh", 0.0)) or False,
    )
    pol = cfg["policy"]
    if pol in ("TTL","RNDTTL"):
        args["ttl"] = int(cfg.get("ttl", 256))
    if pol == "TwoLRU":
        args["filter_ratio"] = float(cfg.get("filter_ratio", 0.1))
    if pol in ("RNDLRU","RNDTTL"):
        args["min_threshold"] = float(cfg.get("min_threshold", 0.5))
        args["max_threshold"] = float(cfg.get("max_threshold", 1.0))
    return args

# -----------------------------
#   GRAFICI (solo matplotlib)
# -----------------------------
def plot_pareto(df_merged, outdir):
    fig = plt.figure()
    plt.scatter(df_merged["cost_per_req"], df_merged["q_mean"])
    plt.xlabel("Costo medio per richiesta")
    plt.ylabel("Qualità media (sim sui hit)")
    plt.title("Frontiera qualità–costo")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "pareto_qualita_costo.png"))
    plt.close(fig)

def plot_hit_rate_vs_threshold(df_merged, outdir):
    if "params" not in df_merged.columns: return
    try:
        thr = df_merged["params"].astype(str).str.extract(r"'threshold': ([0-9\.]+)")[0].astype(float)
    except Exception:
        return
    fig = plt.figure()
    plt.scatter(thr, df_merged["hit_rate"])
    plt.xlabel("Threshold")
    plt.ylabel("Hit rate")
    plt.title("Hit rate vs Threshold")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "hit_rate_vs_threshold.png"))
    plt.close(fig)

def plot_policy_bars(df_merged, outdir):
    # media per policy
    g = df_merged.groupby("policy_name", as_index=False).agg(
        q_mean=("q_mean","mean"),
        hit_rate=("hit_rate","mean"),
        cost_per_req=("cost_per_req","mean"),
    )
    # bar plot qualità media per policy
    fig = plt.figure()
    plt.bar(g["policy_name"], g["q_mean"])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Qualità media (sim hit)")
    plt.title("Qualità media per policy")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "policy_quality_bar.png"))
    plt.close(fig)

    # bar plot hit-rate medio per policy
    fig = plt.figure()
    plt.bar(g["policy_name"], g["hit_rate"])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Hit-rate medio")
    plt.title("Hit-rate medio per policy")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "policy_hitrate_bar.png"))
    plt.close(fig)

    # bar plot costo medio per req per policy
    fig = plt.figure()
    plt.bar(g["policy_name"], g["cost_per_req"])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Costo medio per richiesta")
    plt.title("Costo medio per policy")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "policy_cost_bar.png"))
    plt.close(fig)

def per_run_boxplot_similarity(run_parquet_path, outdir):
    df = pd.read_parquet(run_parquet_path)
    sim_hits = df.loc[df["event"]=="hit","sim"].dropna().to_numpy()
    if sim_hits.size == 0: return
    fig = plt.figure()
    plt.boxplot(sim_hits)
    plt.ylabel("Similarità (hit)")
    plt.title(f"Distribuzione similarità — {pathlib.Path(run_parquet_path).stem}")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{pathlib.Path(run_parquet_path).stem}_sim_hits_box.png"))
    plt.close(fig)

def compute_metrics(df_events, c_lookup=1.0, c_hit=2.0, c_miss=100.0):
    total = len(df_events)
    hits = (df_events["event"] == "hit").sum()
    misses = (df_events["event"] == "miss").sum()
    hit_rate = hits/total if total else 0.0
    sim_hits = df_events.loc[df_events["event"]=="hit","sim"]
    q_mean = float(sim_hits.mean()) if len(sim_hits) else 0.0
    q_p50 = float(sim_hits.quantile(0.50)) if len(sim_hits) else 0.0
    q_p90 = float(sim_hits.quantile(0.90)) if len(sim_hits) else 0.0
    q_p99 = float(sim_hits.quantile(0.99)) if len(sim_hits) else 0.0
    cost = total*c_lookup + hits*c_hit + misses*c_miss
    return dict(total=total, hits=int(hits), misses=int(misses), hit_rate=hit_rate,
                q_mean=q_mean, q_p50=q_p50, q_p90=q_p90, q_p99=q_p99,
                cost_total=cost, cost_per_req=(cost/total if total else 0.0))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Parquet con metadata+CLIP embeddings")
    ap.add_argument("--outdir", default="./outputs", help="Cartella di output")
    ap.add_argument("--runs-jsonl", default=None, help="Opzionale: griglia personalizzata (JSONL). Se assente uso default")
    ap.add_argument("--all", action="store_true", help="Esegui tutte le righe della griglia, altrimenti solo la prima")
    ap.add_argument("--c-lookup", type=float, default=1.0)
    ap.add_argument("--c-hit", type=float, default=2.0)
    ap.add_argument("--c-miss", type=float, default=100.0)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    runs_dir = os.path.join(args.outdir, "runs")
    figs_dir = os.path.join(args.outdir, "figures")
    os.makedirs(runs_dir, exist_ok=True)
    os.makedirs(figs_dir, exist_ok=True)

    # Carica dataset una volta
    mgr = PromptDatasetManager()
    mgr.load_local_metadata(path=args.dataset, load_embeddings=True, add_columns=None)

    # Griglia
    if args.runs_jsonl:
        cfgs = []
        with open(args.runs_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                r = json.loads(line)
                if "dataset_path" not in r: r["dataset_path"] = args.dataset
                cfgs.append(r)
    else:
        cfgs = default_runs_grid(dataset_path=args.dataset)

    registry = build_policy_registry()

    # Costruisci mapping prompt->indice
    prompt_to_idx = {p: i for i, p in enumerate(mgr.prompts_arr)}

    # Esecuzione
    summaries = []
    todo = cfgs if args.all else [cfgs[0]]
    for cfg in todo:
        pol = cfg["policy"]
        if pol not in registry:
            print(f"[WARN] policy {pol} non trovata, skip.")
            continue

        trace_idx = make_trace(mgr, cfg, prompt_to_idx)
        keys = mgr.prompts_arr
        embs = mgr.emb_matrix

        run_id = str(uuid.uuid4())
        sim = CacheSimulator()
        runtime_args = build_runtime_args(cfg, dim=embs.shape[1])
        sim.run(run_id, pol, registry[pol], runtime_args, embs, keys, trace_idx)

        # Salva eventi e summary "raw"
        parquet_path = os.path.join(runs_dir, f"{run_id}.parquet")
        sim.export(path=parquet_path, format="parquet")
        summary = sim.get_summary(run_id)
        summary.update({
            "run_id": run_id,
            "policy_name": pol,
            "params": json.dumps(runtime_args, ensure_ascii=False),
            "backend": cfg.get("backend","faiss_flat"),
            "num_requests": int(cfg.get("num_requests", len(trace_idx))),
            "trace_mode": cfg.get("trace_mode", "sequential"),
            "dataset_path": cfg.get("dataset_path", args.dataset),
        })
        summaries.append(summary)

        # Grafico per-run: boxplot similarità hit
        per_run_boxplot_similarity(parquet_path, figs_dir)

    # Salva summaries complessivi
    df_summ = pd.DataFrame(summaries)
    df_summ.to_csv(os.path.join(args.outdir, "summaries.csv"), index=False)

    # Calcolo metriche derivate + merge
    rows = []
    parquet_paths = list(pathlib.Path(runs_dir).glob("*.parquet"))
    for p in parquet_paths:
        df = pd.read_parquet(p)
        m = compute_metrics(df, c_lookup=args.__dict__["c_lookup"], c_hit=args.__dict__["c_hit"], c_miss=args.__dict__["c_miss"])
        m["run_id"] = p.stem
        rows.append(m)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(os.path.join(args.outdir, "metrics_aggregated.csv"), index=False)

    merged = metrics.merge(df_summ, on="run_id", how="left")
    merged.to_csv(os.path.join(args.outdir, "metrics_with_cfg.csv"), index=False)

    # Grafici cross-run
    plot_pareto(merged, figs_dir)
    plot_hit_rate_vs_threshold(merged, figs_dir)
    plot_policy_bars(merged, figs_dir)

    print("Fatto. Output in:", args.outdir)
    print(" - runs/*.parquet  (eventi per-run)")
    print(" - summaries.csv, metrics_aggregated.csv, metrics_with_cfg.csv")
    print(" - figures/*.png   (grafici per il report)")

if __name__ == "__main__":
    main()