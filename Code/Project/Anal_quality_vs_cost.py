
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
quality_vs_cost.py
==================
Grafici su **Quality (similarità)** vs **Costo Computazionale** (tempo stimato)
dove il costo è derivato da hits/misses con modello:
    estimated_time = hits * HIT_COST + misses * MISS_COST

Caratteristiche chiave
----------------------
- **Costi configurabili** (default globali via CLI) con **override per policy** e **per run**:
  Priorità: per-run > per-policy > default globali.
  * Per-run: colonne `hit_cost`/`miss_cost` in summary.csv **oppure** chiavi in `params/manifest`
             (accetta alias: `hit_time`, `miss_time`).
  * Per-policy: passare mappa tramite `--policy_costs` (JSON o formato semplice, vedi sotto).

- **Per policy**: scatter "Quality vs Estimated Time" con tutte le run della policy.
  * Legenda con **parametri che variano** (no run_id) + opzione `--legend_outside`
  * Optional: **frontiera di Pareto** (min Time, max Quality) con `--pareto`

- **Facet per parametro**: `--facet auto` (o elenco `threshold,ttl`) → un grafico per ogni chiave che varia.

- **Grafico complessivo** raggruppato per policy: `--overall_by_policy`

- **Bar plot** complessivi: `--bars` + `--bars_annotate`
  * Migliore **Quality** per policy
  * **Tempo minimo** per policy

Requisiti plotting
------------------
- Solo Matplotlib (no seaborn)
- Un solo plot per figura
- **Nessun colore specificato** (lasciamo quelli di default). Se passiamo un vettore a `c=`, NON impostiamo cmap.

Esempi
------
Per-policy + Pareto + legenda fuori + costi globali 0.05/5:
    python quality_vs_cost.py --bench_dir ./OUT --hit_cost 0.05 --miss_cost 5 --pareto --legend_outside

Facet automatico:
    python quality_vs_cost.py --bench_dir ./OUT --facet auto --hit_cost 0.05 --miss_cost 5

Grafico complessivo grouped-by-policy:
    python quality_vs_cost.py --bench_dir ./OUT --overall_by_policy

Bar plots complessivi con annotazioni:
    python quality_vs_cost.py --bench_dir ./OUT --bars --bars_annotate

Override costi per policy (JSON):
    python quality_vs_cost.py --bench_dir ./OUT --policy_costs '{"LRUCache":{"hit":0.03,"miss":4},"qLRU":{"hit":0.06,"miss":5}}'

Override costi per policy (formato semplice):
    python quality_vs_cost.py --bench_dir ./OUT --policy_costs "LRUCache:hit=0.03,miss=4; qLRU:hit=0.06,miss=5"
"""

import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------------------- Parsing summary/histories --------------------- #
def _maybe_eval_params(x):
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return {}
        try:
            return json.loads(s)
        except Exception:
            pass
        try:
            return eval(s, {"__builtins__": {}}, {})
        except Exception:
            return {"raw": s}
    return {} if x is None else {"raw": str(x)}


def _read_summary_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    # run_id
    if "run_id" not in df.columns:
        for alt in ["id","run","runid"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "run_id"})
                break
        if "run_id" not in df.columns:
            df["run_id"] = [f"run{idx}" for idx in range(len(df))]

    # policy
    if "policy" not in df.columns:
        for alt in ["algo","class","cache","policy_name"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "policy"})
                break
        if "policy" not in df.columns:
            df["policy"] = "unknown"

    # params
    if "params" in df.columns:
        df["params_dict"] = df["params"].apply(_maybe_eval_params)
    elif "manifest" in df.columns:
        df["params_dict"] = df["manifest"].apply(_maybe_eval_params)
    else:
        df["params_dict"] = [{} for _ in range(len(df))]
    df["param_str"] = df["params_dict"].apply(lambda d: ", ".join(f"{k}={d[k]}" for k in sorted(d)) if isinstance(d, dict) and d else "")

    # metrics
    if "requests" not in df.columns:
        if {"hits","misses"}.issubset(df.columns):
            df["requests"] = df["hits"].fillna(0) + df["misses"].fillna(0)
        else:
            df["requests"] = np.nan
    if "hit_rate" not in df.columns and {"hits","misses"}.issubset(df.columns):
        denom = df["hits"].fillna(0) + df["misses"].fillna(0)
        df["hit_rate"] = np.where(denom>0, df["hits"]/denom, np.nan)

    # quality
    if "quality" not in df.columns:
        for alt in ["avg_similarity","mean_similarity","avg_sim","mean_sim","similarity"]:
            if alt in df.columns:
                df["quality"] = df[alt]
                break
        if "quality" not in df.columns:
            df["quality"] = np.nan

    return df


def _infer_similarity_col(cols):
    for c in ["sim","similarity","score","cosine_sim","dot"]:
        if c in cols:
            return c
    return None


def _read_history_for_run(hist_dir: Path, run_id: str):
    p = hist_dir / f"{run_id}.csv"
    if p.exists():
        try:
            return pd.read_csv(p)
        except Exception:
            return None
    cands = list(hist_dir.glob(f"*{run_id}*.csv"))
    if cands:
        try:
            return pd.read_csv(cands[0])
        except Exception:
            return None
    return None


# --------------------- Helpers & metrics --------------------- #
def _varying_keys(df_policy: pd.DataFrame):
    if "params_dict" not in df_policy.columns:
        return []
    pseries = df_policy["params_dict"].apply(lambda d: d if isinstance(d, dict) else {})
    keys = sorted({k for d in pseries for k in d.keys()})
    varying = []
    for k in keys:
        vals = pseries.apply(lambda d: d.get(k, None))
        if pd.Series(vals).nunique(dropna=True) > 1:
            varying.append(k)
    return varying


def _label_from_params(params: dict, varying_keys: list, fallback_param_str: str = "", fallback_run_id: str = ""):
    # Preferisci solo le chiavi che variano
    if isinstance(params, dict) and varying_keys:
        parts = [f"{k}={params.get(k, None)}" for k in sorted(varying_keys)]
        lbl = ", ".join(parts)
        if lbl.strip():
            return lbl
    # Altrimenti param_str
    if isinstance(fallback_param_str, str) and fallback_param_str.strip():
        return fallback_param_str.strip()
    # fallback: run_id
    if isinstance(fallback_run_id, str) and fallback_run_id.strip():
        return f"{fallback_run_id.strip()}"
    return "run"


def _parse_policy_costs(s: str):
    """
    Accetta JSON o formato semplice:
      "LRUCache:hit=0.03,miss=4; qLRU:hit=0.06,miss=5"
    Ritorna dict: { "LRUCache": {"hit":0.03,"miss":4}, ... }
    """
    if not s:
        return {}
    s = s.strip()
    # Prova JSON
    try:
        obj = json.loads(s)
        # normalizza chiavi interne
        out = {}
        for pol, vals in obj.items():
            if not isinstance(vals, dict):
                continue
            hit = vals.get("hit")
            miss = vals.get("miss")
            # alias
            if hit is None:
                hit = vals.get("hit_cost", vals.get("hit_time"))
            if miss is None:
                miss = vals.get("miss_cost", vals.get("miss_time"))
            try:
                hit = float(hit) if hit is not None else None
                miss = float(miss) if miss is not None else None
            except Exception:
                hit = None if hit is None else float(hit)
                miss = None if miss is None else float(miss)
            out[str(pol)] = {"hit": hit, "miss": miss}
        return out
    except Exception:
        pass

    # Parser formato semplice
    out = {}
    parts = [p for p in s.split(";") if p.strip()]
    for part in parts:
        if ":" not in part:
            continue
        name, kvs = part.split(":", 1)
        name = name.strip()
        hit = None; miss = None
        for kv in kvs.split(","):
            kv = kv.strip()
            if not kv or "=" not in kv:
                continue
            k, v = kv.split("=", 1)
            k = k.strip().lower(); v = v.strip()
            if k in ("hit","hit_cost","hit_time"):
                try: hit = float(v)
                except: pass
            elif k in ("miss","miss_cost","miss_time"):
                try: miss = float(v)
                except: pass
        if name:
            out[name] = {"hit": hit, "miss": miss}
    return out


def _get_costs_for_row(row, default_hit_cost, default_miss_cost, policy_costs_map):
    """
    Precedenza:
      1) per-run: colonne summary (hit_cost/miss_cost) o params_dict['hit_cost'|'miss_cost'|'hit_time'|'miss_time']
      2) per-policy override (policy_costs_map)
      3) default globali
    """
    # per-run dai campi della riga
    hit = row.get("hit_cost", np.nan)
    miss = row.get("miss_cost", np.nan)

    # alias in summary
    if pd.isna(hit):
        hit = row.get("hit_time", np.nan)
    if pd.isna(miss):
        miss = row.get("miss_time", np.nan)

    # per-run dai params
    pdict = row.get("params_dict", {})
    if (pd.isna(hit) or hit is None) and isinstance(pdict, dict):
        hit = pdict.get("hit_cost", pdict.get("hit_time", hit))
    if (pd.isna(miss) or miss is None) and isinstance(pdict, dict):
        miss = pdict.get("miss_cost", pdict.get("miss_time", miss))

    # per-policy override
    pol = str(row.get("policy", "unknown"))
    if (pd.isna(hit) or hit is None) and pol in policy_costs_map:
        val = policy_costs_map[pol].get("hit", None)
        if val is not None:
            hit = val
    if (pd.isna(miss) or miss is None) and pol in policy_costs_map:
        val = policy_costs_map[pol].get("miss", None)
        if val is not None:
            miss = val

    # default globali
    hit = default_hit_cost if (pd.isna(hit) or hit is None) else float(hit)
    miss = default_miss_cost if (pd.isna(miss) or miss is None) else float(miss)
    return float(hit), float(miss)


def _ensure_metrics_cost(row, hist_dir: Path, default_hit_cost: float, default_miss_cost: float, policy_costs_map):
    """Restituisce (quality, estimated_time, hits, misses, hit_cost_used, miss_cost_used)."""
    quality = row.get("quality", np.nan)
    hits = row.get("hits", np.nan)
    misses = row.get("misses", np.nan)

    # fallback da histories
    if (pd.isna(hits) or pd.isna(misses) or pd.isna(quality)) and hist_dir.exists():
        h = _read_history_for_run(hist_dir, str(row["run_id"]))
        if h is not None:
            cols = list(h.columns)
            evt_col = None
            for c in ["event","event_type","evt","type"]:
                if c in cols:
                    evt_col = c
                    break
            if evt_col is not None:
                ev = h[evt_col].astype(str).str.lower()
                hits2 = int((ev == "hit").sum())
                misses2 = int((ev == "miss").sum())
                if pd.isna(hits): hits = hits2
                if pd.isna(misses): misses = misses2
                sim_col = _infer_similarity_col(cols)
                if pd.isna(quality) and sim_col is not None:
                    hit_sims = h.loc[ev=="hit", sim_col]
                    quality = float(np.nanmean(hit_sims)) if len(hit_sims)>0 else float(np.nanmean(h[sim_col]))

    hit_cost_used, miss_cost_used = _get_costs_for_row(row, default_hit_cost, default_miss_cost, policy_costs_map)

    hval = 0 if pd.isna(hits) else float(hits)
    mval = 0 if pd.isna(misses) else float(misses)
    estimated_time = hval * hit_cost_used + mval * miss_cost_used
    return quality, estimated_time, hval, mval, hit_cost_used, miss_cost_used


def _pareto_front_quality_vs_time(points):
    """Pareto frontier per (min time, max quality)."""
    if not points:
        return []
    pts = np.array(points, dtype=float)
    idx = np.argsort(pts[:,0])  # time asc
    pts_sorted = pts[idx]
    frontier = []
    best_q = -np.inf
    for t, q in pts_sorted:
        if q >= best_q:
            frontier.append((t, q))
            best_q = q
    return frontier


# --------------------- Plotting --------------------- #
def _policy_quality_vs_cost(df_policy, hist_dir, out_path, hit_cost, miss_cost, policy_costs_map, legend_outside=False, pareto=True):
    varying = _varying_keys(df_policy)
    xs, ys, labels = [], [], []  # x=time, y=quality

    for _, row in df_policy.iterrows():
        q, t, _, _, _, _ = _ensure_metrics_cost(row, hist_dir, hit_cost, miss_cost, policy_costs_map)
        if pd.isna(q) or pd.isna(t):
            continue
        params = row.get("params_dict", {})
        label = _label_from_params(params, varying, fallback_param_str=row.get("param_str",""), fallback_run_id=str(row.get("run_id","")))
        xs.append(t); ys.append(q); labels.append(label)

    if not xs:
        return False

    plt.figure()
    for x, y, lab in zip(xs, ys, labels):
        plt.scatter([x], [y], label=str(lab))

    if pareto and len(xs) >= 2:
        front = _pareto_front_quality_vs_time(list(zip(xs, ys)))
        if front:
            fx, fy = zip(*front)
            plt.plot(fx, fy, linestyle="-", linewidth=1.5)

    plt.xlabel("estimated_time (s)")
    plt.ylabel("quality")
    polname = str(df_policy["policy"].iloc[0])
    plt.title(f"{polname}: Quality vs Estimated Time")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    # 👉 Annotazione: mostra la migliore run (massimo quality * hit_rate) in alto a destra
    if xs and ys:
        # Trova la migliore run: tempo minimo e qualità massima
        best_idx = None
        best_time = np.inf
        best_quality = -np.inf
        for i, (t, q) in enumerate(zip(xs, ys)):
            if t < best_time or (t == best_time and q > best_quality):
                best_time = t
                best_quality = q
                best_idx = i
                
        if best_idx is not None:
            best_label = labels[best_idx] if labels else ""
            plt.annotate(
                f"Best: {best_label}\nTime={best_time:.3f}, Q={best_quality:.3f}",
                xy=(0.02, 0.98),
                xycoords="axes fraction",
                ha="left",
                va="top",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7)
            )
            # Salva le info dettagliate della migliore run in un file CSV
            csv_path = str(out_path).replace(".png", "_best.csv")
            best_row = df_policy.iloc[[best_idx]].copy()
            best_row.to_csv(csv_path, index=False)
            print(f"✅ Info migliore run salvate in: {csv_path}")

    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()
    return True


def _facet_by_param_quality_vs_cost(df_policy, hist_dir, out_dir, hit_cost, miss_cost, policy_costs_map, params):
    varying = _varying_keys(df_policy)
    if params == "auto":
        keys = varying
    else:
        req = [k.strip() for k in params.split(",") if k.strip()]
        keys = [k for k in req if k in varying]

    for key in keys:
        xs, ys, vals = [], [], []
        for _, row in df_policy.iterrows():
            q, t, _, _, _, _ = _ensure_metrics_cost(row, hist_dir, hit_cost, miss_cost, policy_costs_map)
            if pd.isna(q) or pd.isna(t):
                continue
            v = None
            pdict = row.get("params_dict", {})
            if isinstance(pdict, dict):
                v = pdict.get(key, None)
            xs.append(t); ys.append(q); vals.append(v)

        if not xs:
            continue

        plt.figure()
        # Se i valori sono numerici, passiamo l'array ai colori senza specificare cmap
        floatable = [vv for vv in vals if vv is not None]
        if floatable and all(_is_floatable(vv) for vv in floatable):
            arr = np.array([float(v) if v is not None else np.nan for v in vals], dtype=float)
            m = ~np.isnan(arr)
            plt.scatter(np.array(xs)[m], np.array(ys)[m], c=arr[m])
            if (~m).any():
                plt.scatter(np.array(xs)[~m], np.array(ys)[~m])
        else:
            # discrete legend
            uniq = sorted({str(v) for v in vals})
            for u in uniq:
                mask = np.array([str(v)==u for v in vals])
                plt.scatter(np.array(xs)[mask], np.array(ys)[mask], label=f"{key}={u}")
            plt.legend(title=key, fontsize=8)

        if len(xs) >= 2:
            front = _pareto_front_quality_vs_time(list(zip(xs, ys)))
            if front:
                fx, fy = zip(*front)
                plt.plot(fx, fy, linestyle="-", linewidth=1.5)

        polname = str(df_policy["policy"].iloc[0])
        plt.xlabel("estimated_time (s)"); plt.ylabel("quality")
        plt.title(f"{polname}: Quality vs Estimated Time (by {key})")
        plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        plt.tight_layout()
        out_path = out_dir / f"{polname.replace('/','_').replace(' ','_')}_by_{key}_quality_vs_estimated_time.png"
        plt.savefig(out_path, dpi=160, bbox_inches="tight")
        plt.close()


def _overall_by_policy_quality_vs_cost(df, hist_dir, out_path, hit_cost, miss_cost, policy_costs_map, legend_outside=False):
    if "policy" not in df.columns or df.empty:
        return False

    data = {}
    for pol, g in df.groupby("policy"):
        xs, ys = [], []
        for _, row in g.iterrows():
            q, t, _, _, _, _ = _ensure_metrics_cost(row, hist_dir, hit_cost, miss_cost, policy_costs_map)
            if pd.isna(q) or pd.isna(t):
                continue
            xs.append(t); ys.append(q)
        if xs:
            data[str(pol)] = (xs, ys)

    if not data:
        return False

    plt.figure()
    for pol in sorted(data.keys()):
        xs, ys = data[pol]
        plt.scatter(xs, ys, label=pol)

    plt.xlabel("estimated_time (s)")
    plt.ylabel("quality")
    plt.title("Overall: Quality vs Estimated Time (grouped by policy)")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    if legend_outside:
        plt.legend(title="policy", fontsize=9, loc="center left", bbox_to_anchor=(1.02, 0.5))
    else:
        plt.legend(title="policy", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()
    return True


# --------------------- Bars (best per policy) --------------------- #
def _compute_best_quality_and_lowest_time(df, hist_dir, hit_cost, miss_cost, policy_costs_map):
    rows = []
    if "policy" not in df.columns or df.empty:
        return pd.DataFrame()

    for pol, g in df.groupby("policy"):
        g = g.copy()
        varying = _varying_keys(g)

        best_q = -np.inf; best_q_row = None; best_q_time = np.nan
        best_time = np.inf; best_time_row = None; best_time_quality = np.nan

        for _, row in g.iterrows():
            q, t, _, _, _, _ = _ensure_metrics_cost(row, hist_dir, hit_cost, miss_cost, policy_costs_map)
            if pd.notna(q) and q > best_q:
                best_q = q; best_q_row = row; best_q_time = t
            if pd.notna(t) and t < best_time:
                best_time = t; best_time_row = row
                if pd.notna(q):
                    best_time_quality = q

        def _mk_label(r):
            if r is None:
                return ""
            return _label_from_params(
                r.get("params_dict", {}),
                varying_keys=varying,
                fallback_param_str=r.get("param_str", ""),
                fallback_run_id=str(r.get("run_id",""))
            )

        rows.append({
            "policy": str(pol),
            "best_quality": float(best_q) if best_q != -np.inf else np.nan,
            "best_quality_time": float(best_q_time) if pd.notna(best_q_time) else np.nan,
            "best_quality_label": _mk_label(best_q_row),
            "lowest_time": float(best_time) if best_time != np.inf else np.nan,
            "lowest_time_quality": float(best_time_quality) if pd.notna(best_time_quality) else np.nan,
            "lowest_time_label": _mk_label(best_time_row),
        })

    return pd.DataFrame(rows)


def _bars_best_quality_and_lowest_time(df, hist_dir, out_dir, hit_cost, miss_cost, policy_costs_map, annotate=False):
    out_dir.mkdir(parents=True, exist_ok=True)
    table = _compute_best_quality_and_lowest_time(df, hist_dir, hit_cost, miss_cost, policy_costs_map)
    if table.empty:
        return False

    # Best Quality per policy
    plt.figure()
    plt.bar(table["policy"], table["best_quality"])
    plt.xlabel("policy"); plt.ylabel("best quality")
    plt.title("Overall: Best Quality by Policy")
    plt.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    plt.xticks(rotation=30, ha="right")
    for i, v in enumerate(table["best_quality"]):
        if pd.notna(v):
            plt.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    if annotate:
        for i, (v, lab, t) in enumerate(zip(table["best_quality"], table["best_quality_label"], table["best_quality_time"])):
            if pd.notna(v) and lab:
                s = lab
                if pd.notna(t):
                    s += f"\n(time≈{t:.2f}s)"
                plt.annotate(s, (i, v), xytext=(0, 12), textcoords="offset points", ha="center", va="bottom", fontsize=7)
    plt.tight_layout()
    out1 = out_dir / "overall_best_quality_by_policy.png"
    plt.savefig(out1, dpi=160, bbox_inches="tight")
    plt.close()

    # Lowest Estimated Time per policy
    plt.figure()
    plt.bar(table["policy"], table["lowest_time"])
    plt.xlabel("policy"); plt.ylabel("lowest estimated_time (s)")
    plt.title("Overall: Lowest Estimated Time by Policy")
    plt.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    plt.xticks(rotation=30, ha="right")
    ymax = table["lowest_time"].max() if pd.notna(table["lowest_time"].max()) else 0
    pad = 0.01 * ymax if ymax else 0.01
    for i, v in enumerate(table["lowest_time"]):
        if pd.notna(v):
            plt.text(i, v + pad, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    if annotate:
        for i, (v, lab, q) in enumerate(zip(table["lowest_time"], table["lowest_time_label"], table["lowest_time_quality"])):
            if pd.notna(v) and lab:
                s = lab
                if pd.notna(q):
                    s += f"\n(quality≈{q:.3f})"
                plt.annotate(s, (i, v), xytext=(0, 12), textcoords="offset points", ha="center", va="bottom", fontsize=7)
    plt.tight_layout()
    out2 = out_dir / "overall_lowest_time_by_policy.png"
    plt.savefig(out2, dpi=160, bbox_inches="tight")
    plt.close()

    print(f"✅ Salvati:\n- {out1}\n- {out2}")
    return True


# --------------------- Utils --------------------- #
def _is_floatable(x):
    try:
        float(x)
        return True
    except Exception:
        return False


# --------------------- Main --------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench_dir", type=str, default=".", help="Cartella con summary.csv e histories/")
    ap.add_argument("--out_dir", type=str, default=None, help="Output dir per PNG (default: <bench_dir>/charts_quality_cost)")
    ap.add_argument("--policy", type=str, default="", help="Se impostata, disegna solo questa policy")
    ap.add_argument("--hit_cost", type=float, default=0.05, help="Costo globale di un hit (secondi)")
    ap.add_argument("--miss_cost", type=float, default=5.0, help="Costo globale di un miss (secondi)")
    ap.add_argument("--policy_costs", type=str, default="", help="Override costi per policy (JSON o formato 'Pol:hit=0.03,miss=4; ...')")
    ap.add_argument("--legend_outside", action="store_true", help="Posiziona la legenda fuori dal grafico per-policy")
    ap.add_argument("--pareto", action="store_true", help="Disegna la frontiera di Pareto (min Time, max Quality)")
    ap.add_argument("--facet", type=str, default="", help="Grafici separati per ciascun parametro: 'auto' oppure elenco 'threshold,ttl'")
    ap.add_argument("--overall_by_policy", action="store_true", help="Grafico complessivo grouped by policy (Quality vs Time)")
    ap.add_argument("--bars", action="store_true", help="Bar plot complessivi: best quality e lowest time per policy")
    ap.add_argument("--bars_annotate", action="store_true", help="Annota etichette (parametri) sulle barre")
    args = ap.parse_args()

    bench_dir = Path(args.bench_dir)
    out_dir = Path(args.out_dir) if args.out_dir else (bench_dir / "charts_quality_cost")
    out_dir.mkdir(parents=True, exist_ok=True)
    hist_dir = bench_dir / "histories"

    df = _read_summary_csv(bench_dir / "summary.csv")
    if df.empty:
        print("summary.csv mancante o vuoto.")
        return

    # Prepara mappa costi per policy
    policy_costs_map = _parse_policy_costs(args.policy_costs)

    # Grafico complessivo grouped-by-policy
    overall_path = out_dir / "overall_by_policy_quality_vs_estimated_time.png"
    ok = _overall_by_policy_quality_vs_cost(df, hist_dir, overall_path, args.hit_cost, args.miss_cost, policy_costs_map, legend_outside=args.legend_outside)
    if ok:
        print(f"✅ Salvato: {overall_path}")
    else:
        print("Nessun punto valido per il grafico complessivo grouped-by-policy.")

    # Per-policy
    pols = sorted(df["policy"].dropna().unique()) if "policy" in df.columns else ["unknown"]
    if args.policy:
        pols = [p for p in pols if str(p) == str(args.policy)]
        if not pols:
            print(f"Nessuna run trovata per policy '{args.policy}'.")
            return

    for pol in pols:
        d = df[df["policy"] == pol].copy()
        if d.empty:
            continue
        out_path = out_dir / f"{str(pol).replace('/','_').replace(' ','_')}_quality_vs_estimated_time.png"
        ok = _policy_quality_vs_cost(d, hist_dir, out_path, args.hit_cost, args.miss_cost, policy_costs_map, legend_outside=args.legend_outside, pareto=args.pareto)
        if ok:
            print(f"✅ Salvato: {out_path}")

        if args.facet:
            _facet_by_param_quality_vs_cost(d, hist_dir, out_dir, args.hit_cost, args.miss_cost, policy_costs_map, params=args.facet)

    # Bar plots
    #if args.bars:
    _bars_best_quality_and_lowest_time(df, hist_dir, out_dir, args.hit_cost, args.miss_cost, policy_costs_map, annotate=args.bars_annotate)


if __name__ == "__main__":
    main()
