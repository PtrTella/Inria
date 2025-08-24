# -*- coding: utf-8 -*-
"""
plots_hr_quality.py

HR-vs-Quality plotting utility for similarity-cache benchmarks.
- Un grafico per *ogni threshold* (fissata la *capacity*).
- Asse X = qualità (media coseno sui hit) -> colonna: `avg_similarity`
- Asse Y = hit-rate -> colonna: `hit_rate` (accetta [0,1] o [%])
- Legenda: policy + parametri (se presenti) + costo C_A (+ opzionale hint tempo GPU)
- Evidenzia SOLO la run "migliore" (marker più grande, bordo, annotazione con ★ e prefisso ★ BEST · in legenda).
- Criteri selezione "best": pareto_knee (default), distance, product, hmean, lexi.

Input atteso (cartella benchmark):
- summary.csv    (obbligatorio)
- manifest.json  (opzionale, per leggere num_requests)
- histories/     (non usato direttamente qui)

Colonne attese in summary.csv (se alcune non ci sono, vengono ignorate):
  run_id, policy, hit_rate, miss_rate, avg_similarity, duration_s, C_A,
  capacity, threshold, q, beta, delta, tau, k_duel, max_active_duels, Cr

Uso (programma):
    from plots_hr_quality import plot_all_thresholds
    plot_all_thresholds(
        bench_dir="/path/to/Results/<stamp>",
        capacity=200,
        thresholds=None,          # oppure [0.6, 0.7, 0.8]
        gpu_name=None,            # es. "RTX 4090"
        miss_cost_s=None,         # es. 0.8 sec/img
        hit_cost_s=0.02,          # es. 0.02 sec per hit (lookup/copy)
        save_dir=None,            # default: <bench_dir>/charts_hr_quality
        annotate_best=True,
        best_rule="pareto_knee",
    )

Uso (CLI):
    python -m plots_hr_quality \
      --bench_dir /path/to/Results/<stamp> \
      --capacity 200 \
      --thresholds 0.6 0.7 0.8 \
      --gpu_name "RTX 4090" \
      --miss_cost_s 0.8 \
      --hit_cost_s 0.02 \
      --best_rule pareto_knee

Requisiti: pandas, matplotlib
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------- I/O helpers ----------

def _load_summary(bench_dir: Path) -> pd.DataFrame:
    summary_path = bench_dir / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.csv non trovato in {bench_dir}")
    df = pd.read_csv(summary_path)
    df.columns = [c.strip() for c in df.columns]
    return df


def _load_manifest(bench_dir: Path) -> dict:
    mpath = bench_dir / "manifest.json"
    if mpath.exists():
        with open(mpath, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


# ---------- Label & metrics ----------

def _fmt_params_for_legend(row: pd.Series) -> str:
    pol = str(row.get("policy", ""))
    parts: List[str] = []
    # qLRU-(Delta C)
    if pol.lower().startswith("qlru"):
        if "q" in row and not pd.isna(row["q"]):
            parts.append(f"q={row['q']}")
        # opzionale: Cr se esportato
        if "Cr" in row and not pd.isna(row["Cr"]):
            parts.append(f"Cr={row['Cr']}")
    # Duel
    if pol.lower().startswith("duel"):
        for name, key in [("β","beta"), ("δ","delta"), ("τ","tau"), ("k","k_duel")]:
            if key in row and not pd.isna(row[key]):
                parts.append(f"{name}={row[key]}")
    return ", ".join(parts)


def _normalize_rate(x: float) -> float:
    x = float(x)
    return x / 100.0 if x > 1.0 else x


def _fmt_gpu_time(row: pd.Series,
                  gpu_name: Optional[str],
                  miss_cost_s: Optional[float],
                  hit_cost_s: float,
                  num_requests: Optional[int]) -> str:
    if gpu_name is None or miss_cost_s is None or num_requests is None:
        return ""
    hr = _normalize_rate(row.get("hit_rate", 0.0))
    hits = hr * num_requests
    misses = (1.0 - hr) * num_requests
    est_time = hits * float(hit_cost_s) + misses * float(miss_cost_s)
    t = f"{est_time/60:.1f}m" if est_time >= 60 else f"{est_time:.1f}s"
    return f" (~{t} on {gpu_name})"


def _legend_label(row: pd.Series,
                  gpu_name: Optional[str],
                  miss_cost_s: Optional[float],
                  hit_cost_s: float,
                  num_requests: Optional[int],
                  is_best: bool = False) -> str:
    pol = str(row.get("policy", ""))
    params = _fmt_params_for_legend(row)
    ca_val = row["C_A"] if "C_A" in row and not pd.isna(row["C_A"]) else None
    ca_str = f"CA={ca_val:.3g}" if isinstance(ca_val, (int, float)) else "CA=n/a"
    gpu_hint = _fmt_gpu_time(row, gpu_name, miss_cost_s, hit_cost_s, num_requests)

    base = pol
    if params:
        base += f" ({params})"
    base += f" — {ca_str}{gpu_hint}"
    return base


# ---------- Best selection ----------

def _pareto_front(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    """
    Restituisce i punti non-dominati (massimizza x e y).
    Assume df già normalizzato (y in [0,1]).
    """
    sub = df.copy().sort_values([x_col, y_col], ascending=[False, False])
    out = []
    best_y = -1.0
    for _, r in sub.iterrows():
        y = float(r[y_col])
        if y > best_y:
            out.append(r)
            best_y = y
    return pd.DataFrame(out)


def _choose_best(sub: pd.DataFrame, method: str = "pareto_knee") -> int:
    """
    Ritorna l'indice *iloc* della riga 'migliore' in sub (x=avg_similarity, y=hit_rate).
    Metodi:
      - 'pareto_knee' (default): punto sulla Pareto front con distanza minima da (1,1)
      - 'distance'    : distanza minima euclidea da (1,1)
      - 'product'     : massimizza quality * hit_rate
      - 'hmean'       : massimizza harmonic mean(quality, hit_rate)
      - 'lexi'        : ordina per hit_rate desc, poi quality desc
    """
    subn = sub.reset_index(drop=True).copy()
    # Colonne obbligatorie
    if "avg_similarity" not in subn.columns or "hit_rate" not in subn.columns:
        raise KeyError("summary.csv deve contenere le colonne 'avg_similarity' e 'hit_rate'")

    x = subn["avg_similarity"].astype(float).values
    y = np.array([_normalize_rate(v) for v in subn["hit_rate"].astype(float).values], dtype=float)

    if method == "lexi":
        order = np.lexsort((x * -1.0, y * -1.0))  # lexicographic desc on y then x
        return int(order[0])

    if method == "product":
        scores = x * y
        return int(np.argmax(scores))

    if method == "hmean":
        eps = 1e-12
        scores = 2.0 / (np.maximum(eps, 1.0/x) + np.maximum(eps, 1.0/y))
        return int(np.argmax(scores))

    if method == "pareto_knee":
        pf = _pareto_front(subn.assign(hit_rate_norm=y), "avg_similarity", "hit_rate_norm")
        if len(pf) == 0:
            method = "distance"  # fallback
        else:
            xs = pf["avg_similarity"].astype(float).values
            ys = pf["hit_rate_norm"].astype(float).values
            d2 = (1.0 - xs) ** 2 + (1.0 - ys) ** 2
            best_idx_in_pf = int(np.argmin(d2))
            # L'indice nel df originale è l'index del pf (perché subn.reset_index)
            return int(pf.index[best_idx_in_pf])

    # default: 'distance'
    d2 = (1.0 - x) ** 2 + (1.0 - y) ** 2
    return int(np.argmin(d2))


# ---------- Plotting ----------

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _policy_marker(policy: str) -> str:
    policy = (policy or "").lower()
    if policy == "lru":
        return "o"
    if policy == "lfu":
        return "s"
    if policy.startswith("qlru"):
        return "D"
    if policy.startswith("duel"):
        return "^"
    return "o"


def _filter_capacity_threshold(df: pd.DataFrame, capacity: float, threshold: float) -> pd.DataFrame:
    # tolleranza su float per threshold; capacity spesso è int ma non si sa mai
    df2 = df.copy()
    # numeric
    for col in ("capacity", "threshold"):
        if col in df2.columns:
            df2[col] = pd.to_numeric(df2[col], errors="coerce")
    # filtro
    sel_cap = (df2["capacity"] == capacity) | np.isclose(df2["capacity"], capacity, rtol=0, atol=1e-9)
    sel_thr = (df2["threshold"] == threshold) | np.isclose(df2["threshold"], threshold, rtol=0, atol=1e-9)
    return df2[sel_cap & sel_thr]


def plot_for_threshold(
    df: pd.DataFrame,
    threshold: float,
    capacity: int | float,
    num_requests: Optional[int],
    out_dir: Path,
    gpu_name: Optional[str] = None,
    miss_cost_s: Optional[float] = None,
    hit_cost_s: float = 0.02,
    annotate_best: bool = True,
    best_rule: str = "pareto_knee",
) -> Path:
    sub = _filter_capacity_threshold(df, capacity, threshold)
    if sub.empty:
        raise ValueError(f"Nessuna run trovata per capacity={capacity}, threshold={threshold}")

    # Normalizza indici per selezione best
    sub = sub.reset_index(drop=True)

    # Calcola la run migliore secondo il criterio scelto
    best_iloc = _choose_best(sub, method=best_rule)

    # Plot
    fig, ax = plt.subplots()

    for i, row in sub.iterrows():
        x = float(row["avg_similarity"])
        y = _normalize_rate(row["hit_rate"])
        pol = str(row.get("policy", ""))
        marker = _policy_marker(pol)
        is_best = (i == best_iloc)

        # Stile: best più visibile
        alpha = 1.0 if is_best else 0.7
        ms = 9 if is_best else 6
        mew = 1.6 if is_best else 0.8
        mec = "black" if is_best else None

        ax.plot(
            x, y,
            marker=marker,
            linestyle="",
            markersize=ms,
            markeredgewidth=mew,
            markeredgecolor=mec,
            alpha=alpha,
            label=_legend_label(row, gpu_name, miss_cost_s, hit_cost_s, num_requests, is_best=is_best),
        )

        # Annotazione SOLO per la best
        if annotate_best and is_best:
            ax.annotate(
                f"{pol}\n(q:{x:.3f}, hr:{y:.3f})",
                (x, y),
                xytext=(-110, -10),
                textcoords="offset points",
                fontsize=9,
                fontweight="bold",
            )

    ax.set_xlabel("Quality (mean cosine on hits)")
    ax.set_ylabel("Hit rate")
    ax.set_title(f"Hit Rate vs Quality — capacity={capacity}, threshold={threshold}")
    # legenda a destra, unica
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=14, frameon=False)
    ax.grid(True, linestyle="--", alpha=0.3)

    _ensure_dir(out_dir)
    out_path = out_dir / f"hr_vs_quality_capacity{capacity}_thr{threshold}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_all_thresholds(
    bench_dir: str | Path,
    capacity: int | float,
    thresholds: Optional[Iterable[float]] = None,
    gpu_name: Optional[str] = None,
    miss_cost_s: Optional[float] = None,
    hit_cost_s: float = 0.02,
    save_dir: Optional[str | Path] = None,
    annotate_best: bool = True,
    best_rule: str = "pareto_knee",
) -> List[Path]:
    bench_dir = Path(bench_dir)
    df = _load_summary(bench_dir)
    man = _load_manifest(bench_dir)

    # num_requests per stima tempo GPU (facoltativo)
    num_requests = man.get("num_requests", None)
    try:
        num_requests = int(num_requests) if num_requests is not None else None
    except Exception:
        num_requests = None

    # colonne numeriche coerenti
    for col in ("capacity", "threshold", "avg_similarity", "hit_rate"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # thresholds da plottare
    if thresholds is None:
        thresholds = sorted(df[df["capacity"].round(12) == float(capacity)].dropna(subset=["threshold"])["threshold"].unique().tolist())

    # cartella output
    save_dir = Path(save_dir) if save_dir is not None else (bench_dir / "charts_hr_quality")
    _ensure_dir(save_dir)

    outputs: List[Path] = []
    for thr in thresholds:
        try:
            out = plot_for_threshold(
                df=df,
                threshold=float(thr),
                capacity=float(capacity),
                num_requests=num_requests,
                out_dir=save_dir,
                gpu_name=gpu_name,
                miss_cost_s=miss_cost_s,
                hit_cost_s=hit_cost_s,
                annotate_best=annotate_best,
                best_rule=best_rule,
            )
            outputs.append(out)
        except Exception as e:
            print(f"[warn] threshold={thr}: {e}")
    return outputs


# ---------- CLI ----------

def _parse_cli(argv: Optional[List[str]] = None) -> dict:
    import argparse
    ap = argparse.ArgumentParser(description="Plot HR vs Quality per-threshold per una capacity (con evidenziazione best).")
    ap.add_argument("--bench_dir", required=True, help="Cartella con summary.csv e manifest.json")
    ap.add_argument("--capacity", required=True, type=float, help="Capacity (uguale a quello usato nei run)")
    ap.add_argument("--thresholds", nargs="*", type=float, help="Lista di threshold (default: tutte presenti per quella capacity)")
    ap.add_argument("--gpu_name", type=str, default=None, help="Nome GPU (per annotare costo stimato)")
    ap.add_argument("--miss_cost_s", type=float, default=None, help="Costo miss (sec/img) della GPU")
    ap.add_argument("--hit_cost_s", type=float, default=0.02, help="Costo hit (sec) — lookup/copy")
    ap.add_argument("--save_dir", type=str, default=None, help="Dove salvare i PNG (default: <bench_dir>/charts_hr_quality)")
    ap.add_argument("--best_rule", type=str, default="pareto_knee",
                    choices=["pareto_knee", "distance", "product", "hmean", "lexi"],
                    help="Criterio di scelta del best run")
    ap.add_argument("--no_annotate_best", action="store_true", help="Non annotare il punto migliore")
    args = ap.parse_args(argv)
    return vars(args)


def main():
    args = _parse_cli()
    outs = plot_all_thresholds(
        bench_dir=args["bench_dir"],
        capacity=args["capacity"],
        thresholds=args["thresholds"],
        gpu_name=args["gpu_name"],
        miss_cost_s=args["miss_cost_s"],
        hit_cost_s=args["hit_cost_s"],
        save_dir=args["save_dir"],
        annotate_best=not args["no_annotate_best"],
        best_rule=args["best_rule"],
    )
    if outs:
        print("Salvati:")
        for p in outs:
            print(" -", p)


if __name__ == "__main__":
    bench_dir = "/Users/tella/Workspace/Inria/Results/results_20250823_212138"
    # Read manifest json and run the plot for each capacity
    with open(f"{bench_dir}/manifest.json") as f:
        manifest = json.load(f)
    print(manifest)
    capacities = manifest["args"]["capacities"]
    thresholds = manifest["args"]["thresholds"]
    print(capacities)
    for capacity in capacities:
        plot_all_thresholds(
            bench_dir=bench_dir,
            capacity=capacity,                          # stessa capacity dei run (intero o float, vedi note)
            thresholds=thresholds,                       # oppure [0.6, 0.7, 0.8]
            gpu_name="RTX 4090",                   # opzionale: nome GPU mostrato in legenda
        miss_cost_s=0.8,                       # opzionale: sec/img per miss su quella GPU
        hit_cost_s=0.02                        # opzionale: costo hit (lookup/copy)
    )
