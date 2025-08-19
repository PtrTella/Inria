
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark delle politiche di cache (nessuna UI).
- Carica un dataset parquet con colonne: prompt, clip_emb, user_name, timestamp
- Esegue un grid-search su più politiche con parametri configurabili
- Salva:
  * summary.csv con metriche per ogni run
  * histories/*.csv con la storia evento-per-evento (hit/miss/add/evict)
  * grafici PNG in charts/
Requisiti: numpy, pandas, matplotlib, pyarrow, (faiss/annoy opzionali).
"""

import argparse, json, time, uuid, math, random
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import dei tuoi moduli locali
from PromptDatasetManager import PromptDatasetManager
from BaseCache import CacheSimulator
from CachePolicy import LRUCache, LFUCache, TTLCache
from CacheAware import GreedyCache
from CacheUnAware import QLRUDeltaCCache
from DuelCache_integrated import DuelCache

# --------- Utility ---------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")

def unique_run_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"

def infer_dim(embs: np.ndarray) -> int:
    if embs.ndim != 2:
        raise ValueError(f"Embeddings shape inattesa: {embs.shape}")
    return int(embs.shape[1])

def build_keys(n: int, base: str = "row") -> List[str]:
    # Chiavi univoche per evitare collisioni
    return [f"{base}_{i}" for i in range(n)]

def choose_trace_indices(n_total: int, n_requests: int, mode: str, replace: bool, rng: random.Random) -> np.ndarray:
    n_requests = min(n_requests, n_total) if not replace else n_requests
    if mode == "random":
        idx = rng.choices(range(n_total), k=n_requests) if replace else rng.sample(range(n_total), k=n_requests)
        return np.array(idx, dtype=np.int64)
    else:
        # sequenziale: primi n_requests
        return np.arange(n_requests, dtype=np.int64)

def frequency_from_sequence(keys: List[str]) -> Dict[str, float]:
    # Frequenze relative per Greedy/OSA (se vuoi usarle)
    counts = {}
    for k in keys:
        counts[k] = counts.get(k, 0) + 1
    total = float(sum(counts.values())) or 1.0
    return {k: v/total for k, v in counts.items()}

# --------- Sweep setup ---------

def policy_space(args, dim: int) -> List[Tuple[str, type, Dict]]:
    """
    Restituisce tutte le (policy_name, policy_class, policy_args) da eseguire.
    """
    back = args.backend
    base = dict(dim=dim, backend=back, adaptive_thresh=args.adaptive_factor)
    runs: List[Tuple[str, type, Dict]] = []

    # LRU & LFU
    for C in args.capacities:
        for T in args.thresholds:
            pa = dict(base, capacity=int(C), threshold=float(T))
            if "LRU" in args.policies:
                runs.append(("LRUCache", LRUCache, dict(pa)))
            if "LFU" in args.policies:
                runs.append(("LFUCache", LFUCache, dict(pa)))

    # TTL
    if "TTL" in args.policies:
        for C in args.capacities:
            for T in args.thresholds:
                for ttl in args.ttl_values:
                    pa = dict(base, capacity=int(C), threshold=float(T), ttl=int(ttl))
                    runs.append(("TTLCache", TTLCache, pa))

    # Greedy λ-aware (Neglia et al.)
    if "Greedy" in args.policies:
        # Frequenze uniformi come default; se hai trace keys, puoi ricalcolarle.
        # Nota: Greedy supporta sample_size e rng_seed se definiti nella tua classe.
        for C in args.capacities:
            for T in args.thresholds:
                for cr in args.cr_values:
                    pa = dict(base, capacity=int(C), threshold=float(T), cr=float(cr))
                    runs.append(("GreedyCache", GreedyCache, pa))

    # qLRU-ΔC (Neglia et al.)
    if "qLRUΔC" in args.policies:
        for C in args.capacities:
            for T in args.thresholds:
                for cr in args.cr_values:
                    for q in args.q_values:
                        pa = dict(base, capacity=int(C), threshold=float(T), cr=float(cr), q=float(q))
                        runs.append(("QLRUDeltaCCache", QLRUDeltaCCache, pa))


    # Duel (λ-unaware, counters-based)
    if "Duel" in args.policies:
        for C in args.capacities:
            for T in args.thresholds:
                for beta in args.duel_beta:
                    for delta in args.duel_delta:
                        for tau in args.duel_tau:
                            for kd in args.duel_k:
                                for ma in args.duel_max_active:
                                    for rintf in args.duel_rintf:
                                        pa = dict(base, capacity=int(C), threshold=float(T),
                                                  beta=float(beta), delta=float(delta), tau=int(tau),
                                                  k_duel=int(kd), max_active_duels=int(ma),
                                                  interference_radius=float(rintf))
                                        runs.append(("DuelCache", DuelCache, pa))

    return runs

# --------- Plot helpers (no seaborn, 1 plot per figura, no colori specifici) ---------

def plot_best_hit_by_policy(summary_df: pd.DataFrame, outdir: Path) -> Path:
    best = (summary_df.sort_values("hit_rate", ascending=False)
                      .groupby("policy", as_index=False)
                      .first())
    fig = plt.figure()
    plt.bar(best["policy"], best["hit_rate"])
    plt.title("Miglior hit-rate per policy (su tutta la griglia)")
    plt.ylabel("Hit-rate")
    plt.xticks(rotation=25, ha="right")
    p = outdir / "charts" / "best_hit_rate_by_policy.png"
    ensure_dir(p.parent)
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p

def plot_hit_vs_capacity(summary_df: pd.DataFrame, outdir: Path) -> List[Path]:
    paths = []
    # Per ogni threshold, prendiamo per ciascuna policy la miglior combinazione sugli altri iperparametri
    for th in sorted(summary_df["threshold"].unique()):
        sub = summary_df[summary_df["threshold"] == th].copy()
        # Scegli, per policy+capacity, il run con hit_rate max sugli altri parametri
        pick = (sub.sort_values("hit_rate", ascending=False)
                    .groupby(["policy","capacity"], as_index=False)
                    .first())
        # pivot per grafico
        capacities = sorted(pick["capacity"].unique())
        policies = sorted(pick["policy"].unique())
        fig = plt.figure()
        for pol in policies:
            y = [float(pick[(pick["policy"]==pol)&(pick["capacity"]==c)]["hit_rate"].max()
                      if ((pick["policy"]==pol)&(pick["capacity"]==c)).any() else np.nan) for c in capacities]
            plt.plot(capacities, y, marker="o", label=pol)
        plt.xlabel("Capacity")
        plt.ylabel("Hit-rate (best per policy/capacity)")
        plt.title(f"Hit-rate vs Capacity @ threshold={th}")
        plt.legend()
        p = outdir / "charts" / f"hit_rate_vs_capacity_th{th}.png"
        ensure_dir(p.parent)
        fig.tight_layout()
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)
    return paths

def plot_best_occupancy_curves(summary_df: pd.DataFrame, histories_dir: Path, outdir: Path) -> List[Path]:
    # Per ogni policy, trova il run con miglior hit_rate e plotta l'occupancy nel tempo
    paths = []
    best = (summary_df.sort_values("hit_rate", ascending=False)
                      .groupby("policy", as_index=False)
                      .first())
    for _, row in best.iterrows():
        run_id = row["run_id"]
        hist_path = histories_dir / f"{run_id}.csv"
        if not hist_path.exists():
            continue
        df = pd.read_csv(hist_path)
        # step fittizio = progressivo
        df = df.reset_index().rename(columns={"index":"step"})
        fig = plt.figure()
        plt.plot(df["step"].to_numpy(), df["cache_occupancy"].to_numpy())
        plt.xlabel("Step")
        plt.ylabel("Occupancy")
        plt.title(f"Occupancy nel tempo – best run ({row['policy']})")
        p = outdir / "charts" / f"occupancy_{row['policy']}_{run_id}.png"
        ensure_dir(p.parent)
        fig.tight_layout()
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)
    return paths

# --------- Main ---------

def main():
    ap = argparse.ArgumentParser(description="Benchmark politiche di cache (no UI)")
    ap.add_argument("--data", type=str, default=None, help="Parquet con colonne: prompt, clip_emb, user_name, timestamp")
    ap.add_argument("--outdir", type=str, default=None, help="Cartella risultati (default: results_<timestamp>)")
    ap.add_argument("--max-rows", type=int, default=None, help="Carica al massimo N righe dal dataset")
    ap.add_argument("--num-requests", type=int, default=20000, help="Numero di richieste da simulare")
    ap.add_argument("--trace", choices=["sequential","random"], default="sequential", help="Ordine richieste")
    ap.add_argument("--replace", action="store_true", help="Campionamento random con ripetizione")
    ap.add_argument("--seed", type=int, default=42, help="Seed PRNG per traccia random")

    ap.add_argument("--backend", type=str, default="faiss_flat", help="faiss_flat|faiss_ivf|faiss_hnsw|annoy|linear")
    ap.add_argument("--adaptive-factor", type=float, default=0.0, help="0=off; >0 adatta threshold con occupancy")

    ap.add_argument("--policies", nargs="+",
                    default=["LRU","LFU","qLRUΔC","Duel"],
                    help="Scegli quali policy includere: LRU LFU TTL Greedy qLRUΔC Duel")

    ap.add_argument("--capacities", nargs="+", type=int, default=[50,100,200,500])
    ap.add_argument("--thresholds", nargs="+", type=float, default=[0.7, 0.8])

    # Parametri policy-specifici
    ap.add_argument("--ttl-values", nargs="+", type=int, default=[100, 500, 1000])
    ap.add_argument("--cr-values", nargs="+", type=float, default=[0.2, 0.5, 1.0])
    ap.add_argument("--duel-beta", nargs="+", type=float, default=[0.6,0.75,0.9])
    ap.add_argument("--duel-delta", nargs="+", type=float, default=[0.02,0.05,0.1])
    ap.add_argument("--duel-tau", nargs="+", type=int, default=[100,200,400])
    ap.add_argument("--duel-k", nargs="+", type=int, default=[4,8,16])
    ap.add_argument("--duel-max-active", nargs="+", type=int, default=[8])
    ap.add_argument("--duel-rintf", nargs="+", type=float, default=[0.03])
    ap.add_argument("--q-values", nargs="+", type=float, default=[0.1, 0.2, 0.4])

    ap.add_argument("--synthetic", action="store_true",
                    help="Ignora --data e genera dataset sintetico (per test rapido)")

    args = ap.parse_args()

    outdir = Path(args.outdir) if args.outdir else Path(f"results_{time.strftime('%Y%m%d_%H%M%S')}")
    ensure_dir(outdir)
    histories_dir = outdir / "histories"
    ensure_dir(histories_dir)
    charts_dir = outdir / "charts"
    ensure_dir(charts_dir)

    # --- Caricamento dataset ---
    rng = random.Random(args.seed)
    if args.synthetic:
        N = max(args.num_requests, 5000)
        dim = 512
        # cluster di embedding per rendere plausibili i "hit"
        centers = np.stack([np.random.randn(dim).astype(np.float32) for _ in range(10)], axis=0)
        centers /= np.linalg.norm(centers, axis=1, keepdims=True) + 1e-9
        labels = np.random.choice(len(centers), size=N, replace=True)
        noise = 0.1 * np.random.randn(N, dim).astype(np.float32)
        embs = centers[labels] + noise
        embs /= np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9
        keys = build_keys(N, base="syn")
        # sequenza
        trace_idx = choose_trace_indices(N, args.num_requests, args.trace, args.replace, rng)
    else:
        if not args.data:
            raise SystemExit("--data è obbligatorio (oppure usa --synthetic)")
        mgr = PromptDatasetManager()
        mgr.load_local_metadata(args.data, max_rows=args.max_rows, load_embeddings=True)
        embs = mgr.emb_matrix.astype(np.float32, copy=False)
        dim = infer_dim(embs)
        # Chiavi univoche (evitiamo conflitti sulla stessa stringa prompt)
        keys = build_keys(len(embs), base="row")
        # trace: sequenziale temporale oppure random
        if args.trace == "sequential" and hasattr(mgr, "time_sorted_idx"):
            base_idx = mgr.time_sorted_idx.astype(np.int64)
            if args.num_requests < len(base_idx):
                trace_idx = base_idx[:args.num_requests]
            else:
                trace_idx = base_idx
        else:
            trace_idx = choose_trace_indices(len(embs), args.num_requests, "random", args.replace, rng)

    # --- Costruzione griglia policy ---
    runs = policy_space(args, dim)

    # --- Esecuzione ---
    sim = CacheSimulator()
    summaries = []
    for i, (pol_name, pol_cls, pol_args) in enumerate(runs, start=1):
        # Aggiungi lambda_freq per Greedy se richiesto
        if pol_name == "GreedyCache" and "lambda_freq" not in pol_args:
            # Frequenze uniformi sull'intera sequenza di chiavi della traccia
            freq = {keys[idx]: 1.0 for idx in np.unique(trace_idx)}
            pol_args["lambda_freq"] = freq

        run_id = unique_run_id(f"{pol_name}")
        sim.run(
            run_id=run_id,
            policy_name=pol_name,
            policy_class=pol_cls,
            policy_args=pol_args,
            embeddings=embs,
            keys=keys,
            trace_indices=trace_idx,
        )
        # Salva storia del run
        hist = sim.get_history(run_id)
        hist_path = histories_dir / f"{run_id}.csv"
        hist.to_csv(hist_path, index=False)

        # Riassunto
        meta = sim.get_summary(run_id)
        # Estrai parametri standard noti per comodità (se presenti)
        row = {
            "run_id": meta["run_id"],
            "policy": meta["policy"],
            "hit_rate": float(meta["hit_rate"]),
            "miss_rate": float(meta["miss_rate"]),
            "duration_s": float(meta["duration"]),
            "capacity": pol_args.get("capacity", None),
            "threshold": pol_args.get("threshold", None),
            "ttl": pol_args.get("ttl", None),
            "cr": pol_args.get("cr", None),
            "q": pol_args.get("q", None),
            "backend": pol_args.get("backend", None),
            "adaptive_factor": pol_args.get("adaptive_thresh", None),
        }
        summaries.append(row)

    summary_df = pd.DataFrame(summaries).sort_values(["policy","hit_rate"], ascending=[True,False])
    summary_csv = outdir / "summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    # --- Grafici ---
    chart1 = plot_best_hit_by_policy(summary_df, outdir)
    chart_list2 = plot_hit_vs_capacity(summary_df, outdir)
    chart_list3 = plot_best_occupancy_curves(summary_df, histories_dir, outdir)

    # --- Info finale ---
    manifest = {
        "created_at": now_tag(),
        "num_runs": len(runs),
        "num_requests": int(len(trace_idx)),
        "outdir": str(outdir),
        "artifacts": {
            "summary_csv": str(summary_csv),
            "histories_dir": str(histories_dir),
            "charts": [str(chart1)] + [str(p) for p in chart_list2] + [str(p) for p in chart_list3],
        },
        "args": vars(args),
    }
    with open(outdir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"[OK] Completato. Risultati in: {outdir}")
    print(f" - Summary: {summary_csv}")
    print(f" - Histories: {histories_dir}")
    print(f" - Charts: {outdir/'charts'}")

if __name__ == "__main__":
    main()
