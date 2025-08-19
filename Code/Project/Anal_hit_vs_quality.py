
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
policy_hr_vs_quality_plus.py
----------------------------
Grafici leggibili di Hit Rate vs Quality per POLITICA, con strumenti per capire
subito "quale parametro sta variando".

Funzioni principali:
1) Per ogni policy: un grafico HR vs Quality con tutte le run della policy.
   - legenda = etichette dei parametri che VARIANO (niente run_id)
   - opzionale: legenda spostata fuori (--©)

2) Faceting per PARAMETRO (opzionale, --facet auto o elenco di chiavi):
   - Un grafico per ogni parametro che varia; in ciascuno:
     * se i valori del parametro sono NUMERICI: colore continuo (colormap) + colorbar
     * se CATEGORICI: colori/marker discreti + legenda ordinata

3) Overlay della frontiera di Pareto nel piano (HR, Quality) per la policy (attivabile con --pareto).

Input attesi (in --bench_dir):
- summary.csv con colonne: run_id, policy, params (o manifest), hits/misses o hit_rate, quality (o avg_similarity)
- histories/<run_id>.csv opzionali per derivare hit_rate/quality (da event=hit/miss e sim/similarity)

Uso base (tutte le policy):
    python policy_hr_vs_quality_plus.py --bench_dir ./OUT --out_dir ./OUT/charts

Solo una policy:
    python policy_hr_vs_quality_plus.py --bench_dir ./OUT --out_dir ./OUT/charts --policy LRU

Facet automatico (grafici separati per ogni parametro che varia):
    python policy_hr_vs_quality_plus.py --bench_dir ./OUT --facet auto --legend_outside

Colorare il grafico base per una chiave specifica (es. threshold):
    python policy_hr_vs_quality_plus.py --bench_dir ./OUT --color_by threshold --legend_outside --pareto
"""
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap


# --------------------- I/O & parsing --------------------- #
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

    if "run_id" not in df.columns:
        for alt in ["id", "run", "runid"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "run_id"})
                break
        if "run_id" not in df.columns:
            df["run_id"] = [f"run{idx}" for idx in range(len(df))]

    if "policy" not in df.columns:
        for alt in ["algo", "class", "cache", "policy_name"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "policy"})
                break
        if "policy" not in df.columns:
            df["policy"] = "unknown"

    if "params" in df.columns:
        df["params_dict"] = df["params"].apply(_maybe_eval_params)
    elif "manifest" in df.columns:
        df["params_dict"] = df["manifest"].apply(_maybe_eval_params)
    else:
        df["params_dict"] = [{} for _ in range(len(df))]

    # requests & hit_rate
    if "requests" not in df.columns:
        if "hits" in df.columns and "misses" in df.columns:
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


# --------------------- Helpers --------------------- #
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
    """
    Restituisce SEMPRE una label non-vuota:
    - se esistono chiavi che variano -> "k1=v1, k2=v2"
    - altrimenti prova param_str
    - altrimenti usa run_id
    """
    # Prova con le chiavi che variano
    if isinstance(params, dict) and varying_keys:
        parts = []
        for k in sorted(varying_keys):
            v = params.get(k, None)
            parts.append(f"{k}={v}")
        lbl = ", ".join(parts)
        if lbl.strip():
            return lbl

    # Se niente varia, prova con param_str (se esiste)
    if isinstance(fallback_param_str, str) and fallback_param_str.strip():
        return fallback_param_str.strip()

    # Ultimo fallback: run_id
    if isinstance(fallback_run_id, str) and fallback_run_id.strip():
        return f"{fallback_run_id.strip()}"

    # Estremo: stringa non-vuota generica
    return "run"



def _ensure_metrics(row, hist_dir: Path):
    """Return (hit_rate, quality) for a summary row; fallback to history if needed."""
    hr = row.get("hit_rate", np.nan)
    qt = row.get("quality", np.nan)
    if (pd.isna(hr) or pd.isna(qt)) and hist_dir.exists():
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
                hits = int((ev == "hit").sum())
                misses = int((ev == "miss").sum())
                req = hits + misses
                if pd.isna(hr) and req > 0:
                    hr = hits/req
                sim_col = _infer_similarity_col(cols)
                if pd.isna(qt) and sim_col is not None:
                    hit_sims = h.loc[ev=="hit", sim_col]
                    qt = float(np.nanmean(hit_sims)) if len(hit_sims)>0 else float(np.nanmean(h[sim_col]))
    return hr, qt


def _is_floatable(x):
    try:
        float(x)
        return True
    except Exception:
        return False


def _pareto_front(points):
    """Return points on Pareto frontier (maximize both hr and quality)."""
    if not points:
        return []
    pts = np.array(points, dtype=float)
    order = np.lexsort((-pts[:,1], -pts[:,0]))  # sort: hr desc, quality desc
    pts_sorted = pts[order]
    frontier = []
    best_q = -np.inf
    for hr, q in pts_sorted:
        if q >= best_q:
            frontier.append((hr, q))
            best_q = q
    # unique & sorted
    frontier = sorted(set(frontier), key=lambda t: (-t[0], -t[1]))
    return frontier


# --------------------- Plotting --------------------- #
def _base_policy_plot(df_policy, hist_dir, out_path, legend_outside=False, color_by=None, pareto=True):
    """Un plot per policy: un punto per run; legenda dalle chiavi che variano.
       Se color_by è numerico, usa colormap + colorbar.
    """
    varying = _varying_keys(df_policy)
    xs, ys, labels = [], [], []
    color_vals = []

    for _, row in df_policy.iterrows():
        hr, qt = _ensure_metrics(row, hist_dir)
        if pd.isna(hr) or pd.isna(qt):
            continue

        params = row.get("params_dict", {})
        param_str = row.get("param_str", "")
        run_id = str(row.get("run_id", ""))

        label = _label_from_params(params, varying, fallback_param_str=param_str, fallback_run_id=run_id)

        xs.append(hr); ys.append(qt); labels.append(label)

        if color_by:
            val = None
            if isinstance(params, dict):
                val = params.get(color_by, None)
            color_vals.append(val)

    if not xs:
        return False

    plt.figure()
    used_colorbar = False

    if color_by and len(color_vals) == len(xs) and all(_is_floatable(v) for v in color_vals if v is not None):
        # Colorbar continua per parametri numerici
        cv = np.array([float(v) if v is not None else np.nan for v in color_vals], dtype=float)
        mask = ~np.isnan(cv)
        plt.scatter(np.array(xs)[mask], np.array(ys)[mask], c=cv[mask], cmap=get_cmap("viridis"), alpha=0.9)
        if (~mask).any():
            plt.scatter(np.array(xs)[~mask], np.array(ys)[~mask], alpha=0.6)
        cb = plt.colorbar()
        cb.set_label(color_by)
        used_colorbar = True
    else:
        # Un punto per run con etichetta NON VUOTA
        for x, y, lab in zip(xs, ys, labels):
            if not lab or not str(lab).strip():
                lab = "run"
            plt.scatter([x], [y], label=str(lab), alpha=0.9)

    # Pareto overlay (facoltativo: lascialo com'era)
    if pareto and len(xs) >= 2:
        front = _pareto_front(list(zip(xs, ys)))
        if front:
            fx, fy = zip(*front)
            plt.plot(fx, fy, linestyle="-", linewidth=1.5)

    plt.xlabel("hit_rate"); plt.ylabel("quality")
    polname = str(df_policy["policy"].iloc[0])
    ttl = polname + ": Hit Rate vs Quality"
    if color_by:
        ttl += f"  (color: {color_by})"
    plt.title(ttl)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.xlim(0, 1); plt.ylim(0, 1)
    
    # 👉 Annotazione: mostra la migliore run (massimo quality * hit_rate) in alto a destra
    if xs and ys:
        scores = [x * y for x, y in zip(xs, ys)]
        best_idx = int(np.argmax(scores))
        best_label = labels[best_idx] if labels else ""
        best_hr = xs[best_idx]
        best_qt = ys[best_idx]
        plt.annotate(
            f"Best: {best_label}\nHR={best_hr:.3f}, Q={best_qt:.3f}",
            xy=(0.98, 0.98),
            xycoords="axes fraction",
            ha="right",
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



def _facet_by_param(df_policy, hist_dir, out_dir, params):
    """Grafici separati per ciascun parametro indicato (o auto)."""
    varying = _varying_keys(df_policy)
    if params == "auto":
        keys = varying
    else:
        req = [k.strip() for k in params.split(",") if k.strip()]
        keys = [k for k in req if k in varying]

    for key in keys:
        xs, ys, vals = [], [], []
        for _, row in df_policy.iterrows():
            hr, qt = _ensure_metrics(row, hist_dir)
            if pd.isna(hr) or pd.isna(qt):
                continue
            v = None
            pdict = row.get("params_dict", {})
            if isinstance(pdict, dict):
                v = pdict.get(key, None)
            xs.append(hr); ys.append(qt); vals.append(v)

        if not xs:
            continue

        plt.figure()
        numeric = all(_is_floatable(v) for v in vals if v is not None)
        if numeric:
            import numpy as _np
            _vv = _np.array([float(v) if v is not None else _np.nan for v in vals], dtype=float)
            m = ~_np.isnan(_vv)
            sc = plt.scatter(_np.array(xs)[m], _np.array(ys)[m], c=_vv[m], cmap=get_cmap("viridis"), alpha=0.9)
            if (~m).any():
                plt.scatter(_np.array(xs)[~m], _np.array(ys)[~m], alpha=0.6)
            cb = plt.colorbar(sc)
            cb.set_label(key)
        else:
            unique_vals = sorted({str(v) for v in vals})
            for uv in unique_vals:
                mask = np.array([str(v) == uv for v in vals])
                plt.scatter(np.array(xs)[mask], np.array(ys)[mask], label=f"{key}={uv}", alpha=0.9)
            plt.legend(title=key, fontsize=8)

        if len(xs) >= 2:
            front = _pareto_front(list(zip(xs, ys)))
            if front:
                fx, fy = zip(*front)
                plt.plot(fx, fy, linestyle="-", linewidth=1.5)

        plt.xlabel("hit_rate"); plt.ylabel("quality")
        polname = str(df_policy["policy"].iloc[0])
        plt.title(f"{polname}: Hit Rate vs Quality (by {key})")
        plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        plt.xlim(0,1); plt.ylim(0,1)
        out_path = out_dir / f"{polname.replace('/','_').replace(' ','_')}_by_{key}_hit_rate_vs_quality.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=160, bbox_inches="tight")
        plt.close()


# --------------- Overall Policy --------------- #

def _overall_by_policy_plot(df, hist_dir, out_path, legend_outside=False):
    """Grafico unico: Hit Rate vs Quality per tutte le policy, raggruppato per policy."""
    if "policy" not in df.columns or df.empty:
        return False

    xs_by_pol = {}
    ys_by_pol = {}

    for pol, g in df.groupby("policy"):
        xs, ys = [], []
        for _, row in g.iterrows():
            hr, qt = _ensure_metrics(row, hist_dir)
            if pd.isna(hr) or pd.isna(qt):
                continue
            xs.append(hr)
            ys.append(qt)
        if xs:
            xs_by_pol[str(pol)] = xs
            ys_by_pol[str(pol)] = ys

    if not xs_by_pol:
        return False

    import matplotlib.pyplot as plt
    plt.figure()

    # Un set di punti per ogni policy (Matplotlib gestisce i colori in automatico)
    for pol in sorted(xs_by_pol.keys()):
        plt.scatter(xs_by_pol[pol], ys_by_pol[pol], label=pol, alpha=0.9)

    plt.xlabel("hit_rate")
    plt.ylabel("quality")
    plt.title("Overall: Hit Rate vs Quality (grouped by policy)")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    # Legenda dentro o fuori a scelta
    if legend_outside:
        plt.legend(title="policy", fontsize=9, loc="center left", bbox_to_anchor=(1.02, 0.5))
    else:
        plt.legend(title="policy", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()
    return True


def _compute_best_per_policy(df, hist_dir):
    """
    Per ogni policy, seleziona:
      - la run con Hit Rate massimo
      - la run con Quality (similarità) massima
    Restituisce un DataFrame con colonne:
      policy, best_hit_rate, best_hr_label, best_quality, best_q_label
    """
    rows = []
    if "policy" not in df.columns or df.empty:
        return pd.DataFrame()

    for pol, g in df.groupby("policy"):
        g = g.copy()
        # quali chiavi variano in questa policy (per creare label pulite)
        varying = _varying_keys(g)

        best_hr = -np.inf
        best_hr_row = None
        best_q = -np.inf
        best_q_row = None

        for _, row in g.iterrows():
            hr, qt = _ensure_metrics(row, hist_dir)
            # max hit rate
            if pd.notna(hr) and hr > best_hr:
                best_hr = hr
                best_hr_row = row
            # max quality
            if pd.notna(qt) and qt > best_q:
                best_q = qt
                best_q_row = row

        # prepara etichette leggibili per le run vincenti
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
            "best_hit_rate": float(best_hr) if best_hr != -np.inf else np.nan,
            "best_hr_label": _mk_label(best_hr_row),
            "best_quality": float(best_q) if best_q != -np.inf else np.nan,
            "best_q_label": _mk_label(best_q_row),
        })

    return pd.DataFrame(rows)


def _bar_best_per_policy(df, hist_dir, out_dir, legend_outside=False, annotate_labels=False):
    """
    Crea due bar plot:
      - overall_best_hit_rate_by_policy.png
      - overall_best_quality_by_policy.png
    Ogni barra = migliore run della policy per la metrica in questione.
    annotate_labels=True: scrive l’etichetta della config vincente a fianco/ sopra la barra.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    table = _compute_best_per_policy(df, hist_dir)
    if table.empty:
        return False

    # --- Bar: Best Hit Rate ---
    plt.figure()
    x = np.arange(len(table))
    plt.bar(table["policy"], table["best_hit_rate"])
    plt.xlabel("policy")
    plt.ylabel("best hit_rate")
    plt.title("Overall: Best Hit Rate by Policy")
    plt.ylim(0, 1)
    plt.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    plt.xticks(rotation=30, ha="right")

    # annotazioni valore numerico
    for i, v in enumerate(table["best_hit_rate"]):
        if pd.notna(v):
            plt.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    # etichette (parametri) opzionali
    if annotate_labels:
        for i, (v, lab) in enumerate(zip(table["best_hit_rate"], table["best_hr_label"])):
            if pd.notna(v) and lab:
                plt.annotate(lab, (i, v), xytext=(0, 12), textcoords="offset points",
                             ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    out1 = out_dir / "overall_best_hit_rate_by_policy.png"
    plt.savefig(out1, dpi=160, bbox_inches="tight")
    plt.close()

    # --- Bar: Best Quality ---
    plt.figure()
    x = np.arange(len(table))
    plt.bar(table["policy"], table["best_quality"])
    plt.xlabel("policy")
    plt.ylabel("best quality")
    plt.title("Overall: Best Quality by Policy")
    plt.ylim(0, 1)
    plt.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    plt.xticks(rotation=30, ha="right")

    # annotazioni valore numerico
    for i, v in enumerate(table["best_quality"]):
        if pd.notna(v):
            plt.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    # etichette (parametri) opzionali
    if annotate_labels:
        for i, (v, lab) in enumerate(zip(table["best_quality"], table["best_q_label"])):
            if pd.notna(v) and lab:
                plt.annotate(lab, (i, v), xytext=(0, 12), textcoords="offset points",
                             ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    out2 = out_dir / "overall_best_quality_by_policy.png"
    plt.savefig(out2, dpi=160, bbox_inches="tight")
    plt.close()

    print(f"✅ Salvati:\n- {out1}\n- {out2}")
    return True


# --------------------- Main --------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench_dir", type=str, default=".", help="Cartella con summary.csv e histories/")
    ap.add_argument("--out_dir", type=str, default="", help="Output dir per PNG (default: <bench_dir>/charts)")
    ap.add_argument("--policy", type=str, default="", help="Se impostata, disegna solo questa policy")
    ap.add_argument("--legend_outside", default=True, action="store_true", help="Posiziona la legenda fuori dal grafico")
    ap.add_argument("--pareto", action="store_true", help="Disegna la frontiera di Pareto (HR,Quality)")
    ap.add_argument("--color_by", type=str, default="", help="Colora il grafico base per il valore di questa chiave (numeric->colorbar, categorical->legend)")
    ap.add_argument("--facet", type=str, default="", help="Crea grafici separati per ciascun parametro: 'auto' oppure elenco 'threshold,ttl'")
    args = ap.parse_args()

    bench_dir = Path(args.bench_dir)
    out_dir = Path(args.out_dir) if args.out_dir else (bench_dir / "charts")
    out_dir.mkdir(parents=True, exist_ok=True)
    hist_dir = bench_dir / "histories"

    df = _read_summary_csv(bench_dir / "summary.csv")
    if df.empty:
        print("summary.csv mancante o vuoto.")
        return

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
        out_path = out_dir / f"{str(pol).replace('/','_').replace(' ','_')}_hit_rate_vs_quality.png"
        ok = _base_policy_plot(d, hist_dir, out_path, legend_outside=args.legend_outside, color_by=(args.color_by or None), pareto=args.pareto)
        if ok:
            print(f"✅ Salvato: {out_path}")
        else:
            print(f"Policy '{pol}': nessun punto valido.")

        if args.facet:
            _facet_by_param(d, hist_dir, out_dir, params=args.facet)

    ok = _overall_by_policy_plot(df, hist_dir, out_dir / "overall_hit_rate_vs_quality.png", legend_outside=args.legend_outside)
    if ok:
        print(f"✅ Salvato: {out_dir / 'overall_hit_rate_vs_quality.png'}")
    else:
        print("Nessun punto valido per il grafico complessivo.")


    _bar_best_per_policy(df, hist_dir, out_dir, legend_outside=args.legend_outside, annotate_labels=True)

if __name__ == "__main__":
    main()
