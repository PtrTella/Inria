#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json, math
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def read_summary(results_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(results_dir / "summary.csv")
    if "duration_s" not in df.columns and "duration" in df.columns:
        df["duration_s"] = df["duration"]
    return df

def read_history(results_dir: Path, run_id: str) -> pd.DataFrame:
    return pd.read_csv(results_dir / "histories" / f"{run_id}.csv")

def binom_ci_95(p: float, n: int) -> Tuple[float,float]:
    if n <= 0: return (np.nan, np.nan)
    se = math.sqrt(max(p*(1.0-p), 1e-12) / n)
    return (max(0.0, p - 1.96*se), min(1.0, p + 1.96*se))

COST_PRESETS = {
    "a100_pytorch_sdxl": {"hit_ms": 50, "miss_ms": 9000},
    "a100_tensorrt_sdxl": {"hit_ms": 50, "miss_ms": 3000},
    "l40s_replicate_sdxl": {"hit_ms": 60, "miss_ms": 5000},
    "rtx4090_sd15_1024": {"hit_ms": 50, "miss_ms": 4500},
    "consumer_default": {"hit_ms": 80, "miss_ms": 7000},
}

def effective_latency_ms(hit_rate: float, hit_ms: float, miss_ms: float) -> float:
    return hit_rate*hit_ms + (1.0-hit_rate)*miss_ms

def pareto_frontier(points: List[Tuple[float,float]]) -> List[Tuple[float,float]]:
    pts = sorted(points, key=lambda t: (t[1], -t[0]))
    frontier, best_x = [], -1.0
    for x,y in pts:
        if x > best_x:
            frontier.append((x,y)); best_x = x
    return frontier

def attach_request_counts(summary: pd.DataFrame, results_dir: Path) -> pd.DataFrame:
    n_list = []
    for rid in summary["run_id"]:
        p = results_dir / "histories" / f"{rid}.csv"
        if p.exists():
            try:
                n_list.append(sum(1 for _ in open(p, "r", encoding="utf-8")) - 1)
            except Exception:
                n_list.append(np.nan)
        else:
            n_list.append(np.nan)
    summary = summary.copy()
    summary["num_requests"] = n_list
    return summary

def compute_best_by_policy(summary: pd.DataFrame) -> pd.DataFrame:
    return (summary.sort_values("hit_rate", ascending=False).groupby("policy", as_index=False).first())

def add_cost_model(summary: pd.DataFrame, hit_ms: float, miss_ms: float) -> pd.DataFrame:
    df = summary.copy()
    df["est_latency_ms"] = df["hit_rate"].apply(lambda p: effective_latency_ms(p, hit_ms, miss_ms))
    df["efficiency_index"] = df["hit_rate"] / (df["est_latency_ms"]/1000.0 + 1e-9)
    return df

def sensitivity_sweep(summary: pd.DataFrame, hit_ms: float, miss_ms_grid: List[int]) -> pd.DataFrame:
    rows = []
    for miss_ms in miss_ms_grid:
        tmp = add_cost_model(summary, hit_ms, miss_ms).assign(miss_ms=miss_ms)
        rows.append(tmp[["run_id","policy","capacity","threshold","ttl","cr","q","hit_rate","est_latency_ms","miss_ms"]])
    return pd.concat(rows, axis=0, ignore_index=True)

def agg_by_policy_capacity_threshold(df: pd.DataFrame, value: str) -> pd.DataFrame:
    g = (df.groupby(["policy","capacity","threshold"], as_index=False)[value].agg(["mean","std","count"]).reset_index())
    g.columns = ["policy","capacity","threshold","mean","std","count"]
    return g

def plot_bar_with_ci(df: pd.DataFrame, out_p: Path, title: str, xcol: str, ycol: str, ncol: Optional[str]=None):
    fig = plt.figure()
    x = df[xcol].astype(str).tolist()
    y = df[ycol].to_numpy()
    if ncol is not None:
        cis = [binom_ci_95(float(yy), int(nn)) for yy, nn in zip(y, df[ncol].to_numpy())]
        lo = [max(0.0, y[i]-cis[i][0]) for i in range(len(y))]
        hi = [max(0.0, cis[i][1]-y[i]) for i in range(len(y))]
        plt.bar(x, y, yerr=[lo,hi], capsize=4)
    else:
        plt.bar(x, y)
    plt.title(title); plt.xticks(rotation=25, ha="right"); plt.ylabel(ycol)
    plt.tight_layout(); plt.savefig(out_p, dpi=150); plt.close(fig)

def plot_lines_by_policy(df: pd.DataFrame, out_p: Path, title: str, xcol: str, ycol: str):
    fig = plt.figure()
    for pol in sorted(df["policy"].unique()):
        sub = df[df["policy"]==pol].sort_values(xcol)
        plt.plot(sub[xcol].to_numpy(), sub[ycol].to_numpy(), marker="o", label=pol)
    plt.title(title); plt.xlabel(xcol); plt.ylabel(ycol); plt.legend()
    plt.tight_layout(); plt.savefig(out_p, dpi=150); plt.close(fig)

def plot_scatter_with_frontier(points_df: pd.DataFrame, out_p: Path, title: str):
    fig = plt.figure()
    plt.scatter(points_df["hit_rate"].to_numpy(), points_df["est_latency_ms"].to_numpy())
    pts = list(zip(points_df["hit_rate"].to_numpy(), points_df["est_latency_ms"].to_numpy()))
    front = pareto_frontier(pts)
    if len(front) >= 2:
        fx = [t[0] for t in front]; fy = [t[1] for t in front]
        plt.plot(fx, fy, marker="o")
    plt.title(title); plt.xlabel("Hit-rate (↑ meglio)"); plt.ylabel("Latency stimata (ms, ↓ meglio)")
    plt.tight_layout(); plt.savefig(out_p, dpi=150); plt.close(fig)

def plot_hist_similarity(df: pd.DataFrame, out_p: Path, title: str):
    fig = plt.figure(); plt.hist(df["similarity"].to_numpy(), bins=50)
    plt.title(title); plt.xlabel("similarity"); plt.ylabel("count")
    plt.tight_layout(); plt.savefig(out_p, dpi=150); plt.close(fig)

def plot_rolling_hit_ratio(hist: pd.DataFrame, out_p: Path, title: str, window: int = 200):
    if "is_hit" not in hist.columns: return
    fig = plt.figure()
    s = pd.Series(hist["is_hit"].astype(int).to_numpy())
    r = s.rolling(window, min_periods=max(5, window//5)).mean()
    plt.plot(np.arange(len(s)), r.to_numpy())
    plt.title(title); plt.xlabel("step"); plt.ylabel(f"rolling_hit_rate@{window}")
    plt.tight_layout(); plt.savefig(out_p, dpi=150); plt.close(fig)

def main():
    ap = argparse.ArgumentParser(description="Analisi avanzata risultati cache")
    ap.add_argument("--results-dir", type=str, required=True)
    ap.add_argument("--cost-preset", type=str, default="a100_tensorrt_sdxl", choices=list(COST_PRESETS.keys()))
    ap.add_argument("--hit-ms", type=float, default=None)
    ap.add_argument("--miss-ms", type=float, default=None)
    ap.add_argument("--sweep-miss", nargs="+", type=int, default=[1000,2000,3000,5000,7000,9000])
    ap.add_argument("--max-histories", type=int, default=40)
    args = ap.parse_args()

    results_dir = Path(args.results_dir); outdir = results_dir / "analysis_outputs"; ensure_dir(outdir)
    summary = read_summary(results_dir); summary = attach_request_counts(summary, results_dir)

    preset = COST_PRESETS[args.cost_preset].copy()
    if args.hit_ms is not None: preset["hit_ms"] = float(args.hit_ms)
    if args.miss_ms is not None: preset["miss_ms"] = float(args.miss_ms)
    with open(outdir / "cost_preset_used.json", "w", encoding="utf-8") as f: json.dump(preset, f, indent=2, ensure_ascii=False)

    scored = add_cost_model(summary, preset["hit_ms"], preset["miss_ms"])
    best_by_pol = compute_best_by_policy(scored)

    plot_bar_with_ci(best_by_pol.assign(policy_lbl=lambda d: d["policy"]),
                     outdir / "best_hit_rate_by_policy_CI.png",
                     "Miglior hit-rate per policy (CI 95%)",
                     "policy","hit_rate","num_requests")

    for th in sorted(scored["threshold"].dropna().unique()):
        sub = scored[scored["threshold"]==th].copy()
        pick = (sub.sort_values("hit_rate", ascending=False).groupby(["policy","capacity"], as_index=False).first())
        plot_lines_by_policy(pick.sort_values(["policy","capacity"]),
                             outdir / f"hit_rate_vs_capacity_th{th}.png",
                             f"Hit-rate vs Capacity @ threshold={th} (best sugli altri parametri)",
                             "capacity","hit_rate")
        pickL = (sub.sort_values("est_latency_ms", ascending=True).groupby(["policy","capacity"], as_index=False).first())
        plot_lines_by_policy(pickL.sort_values(["policy","capacity"]),
                             outdir / f"est_latency_ms_vs_capacity_th{th}.png",
                             f"Latenza stimata vs Capacity @ threshold={th} (best sugli altri parametri)",
                             "capacity","est_latency_ms")

    plot_scatter_with_frontier(scored[["hit_rate","est_latency_ms"]].copy(),
                               outdir / "scatter_hit_vs_latency_pareto.png",
                               "Trade-off: hit-rate vs latenza stimata (Pareto frontier)")

    plot_bar_with_ci(best_by_pol, outdir / "efficiency_index_best_by_policy.png",
                     "Indice di efficienza (hit-rate / sec) – best per policy",
                     "policy","efficiency_index", None)

    sens = sensitivity_sweep(scored, preset["hit_ms"], args.sweep_miss)
    g = (sens.groupby(["policy","miss_ms"], as_index=False)["est_latency_ms"].mean())
    for pol in sorted(g["policy"].unique()):
        sub = g[g["policy"]==pol].sort_values("miss_ms")
        fig = plt.figure(); plt.plot(sub["miss_ms"].to_numpy(), sub["est_latency_ms"].to_numpy(), marker="o")
        plt.title(f"Sensitività costo miss – {pol}"); plt.xlabel("miss_ms"); plt.ylabel("latency media stimata (ms)")
        plt.tight_layout(); plt.savefig(outdir / f"sensitivity_latency_vs_miss_ms_{pol}.png", dpi=150); plt.close(fig)

    # per-run dettagli (rolling hit, occupancy, similarity)
    chosen_runs = best_by_pol["run_id"].tolist()
    if len(chosen_runs) < args.max_histories:
        extra = [rid for rid in scored["run_id"].tolist() if rid not in chosen_runs]
        chosen_runs.extend(extra[:max(0, args.max_histories - len(chosen_runs))])

    for rid in chosen_runs:
        p = results_dir / "histories" / f"{rid}.csv"
        if not p.exists(): continue
        hist = pd.read_csv(p)
        plot_rolling_hit_ratio(hist, outdir / f"rolling_hit_rate_{rid}.png", f"Rolling hit-rate – {rid}",
                               window=max(50, min(500, int(len(hist)*0.05))))
        if "cache_occupancy" in hist.columns:
            fig = plt.figure(); plt.plot(np.arange(len(hist)), hist["cache_occupancy"].to_numpy())
            plt.title(f"Occupancy nel tempo – {rid}"); plt.xlabel("step"); plt.ylabel("occupancy")
            plt.tight_layout(); plt.savefig(outdir / f"occupancy_{rid}.png", dpi=150); plt.close(fig)
        if "similarity" in hist.columns:
            plot_hist_similarity(hist, outdir / f"similarity_hist_{rid}.png", f"Distribuzione similarity – {rid}")
            if "is_hit" in hist.columns:
                plot_hist_similarity(hist[hist["is_hit"]==1], outdir / f"similarity_hist_hits_{rid}.png",
                                     f"Distribuzione similarity (HIT) – {rid}")
                plot_hist_similarity(hist[hist["is_hit"]==0], outdir / f"similarity_hist_miss_{rid}.png",
                                     f"Distribuzione similarity (MISS) – {rid}")

    best_pc = (scored.sort_values("hit_rate", ascending=False).groupby(["policy","capacity"], as_index=False).first())
    best_pc.to_csv(outdir / "best_by_policy_capacity.csv", index=False)

    agg = (scored.groupby(["policy","capacity","threshold"], as_index=False)["hit_rate"].agg(["mean","std","count"]).reset_index())
    agg.columns = ["policy","capacity","threshold","mean","std","count"]
    agg.to_csv(outdir / "agg_hit_rate_by_pct.csv", index=False)

    agg_lat = (scored.groupby(["policy","capacity","threshold"], as_index=False)["est_latency_ms"].agg(["mean","std","count"]).reset_index())
    agg_lat.columns = ["policy","capacity","threshold","mean","std","count"]
    agg_lat.to_csv(outdir / "agg_latency_by_pct.csv", index=False)

    print("[OK] Analisi completata."); print("Output:", outdir)

if __name__ == "__main__":
    main()
