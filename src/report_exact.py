"""
Display the results of the exact-stream pipeline (src/train_exact.py).
Reads saved result files only - nothing is trained, changed or re-scored.

  1. Subtype classification - our three bagged outputs (transformer stream, graph stream,
     Ensembled Transformer-Graph Streams; 5-fold mean +- std) next to the values reported in the base papers
     (configs/base_paper_results.json, copied from the PDFs).
  2. Survival prediction - C-index of the Ensembled Transformer-Graph Streams; out-of-fold risk scores split at
     the median of each test fold into high / low risk; Kaplan-Meier curves, log-rank
     test, Cox hazard ratio (high vs low) and 5-year survival per group.
  3. Biological interpretation - only for cohorts trained with --interpret
     (<cohort>_feature_importance.csv, reference transformer paper Sec. 2.5 method, src/interpret_exact.py).
     Importance of a feature = its performance drop averaged over ALL bagging members of
     all folds (a member that did not use the feature counts as 0 drop). Top 30 features
     per omics for subtype (Ensembled Transformer-Graph Streams and transformer stream alone) and for survival, whole-omics
     importance, and gene / miRNA lists for GO / KEGG enrichment.

Output: results/<tag>/report/   and   results/figures/<tag>/
Usage:  python -m src.report_exact --cohort all [--tag final] [--results_dir DIR --out_dir DIR]
Cohorts come from configs/cohorts.json; cohorts without results are skipped.
"""

import os
import json
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from lifelines.plotting import add_at_risk_counts

OUTS = [("trans", "Our transformer stream (bagged)"), ("graph", "Our graph stream (bagged)"),
        ("ensembled", "Our Ensembled Transformer-Graph Streams model (bagged)")]
METRICS = [("accuracy", "Accuracy"), ("weighted_f1", "Weighted-F1"), ("macro_f1", "Macro-F1")]
PAPERS = "configs/base_paper_results.json"
TOP = 30


def fmt(mean, std=None):
    if mean is None or (isinstance(mean, float) and np.isnan(mean)):
        return "-"
    return f"{mean:.3f}" if std is None else f"{mean:.3f} ± {std:.3f}"


def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 1. classification
# ---------------------------------------------------------------------------
def classification_table(cohort, res):
    papers = json.load(open(PAPERS))
    # a cohort variant (e.g. brca_pancan) is compared with the paper values of its base cohort ("paper_key")
    key = json.load(open("configs/cohorts.json")).get(cohort, {}).get("paper_key", cohort)
    rows = []
    for method, entry in papers.items():
        if method.startswith("_"):
            continue
        p = entry.get(key)
        if p is None:
            rows.append({"Method": f"{method} (paper)", "Accuracy": "-", "Weighted-F1": "-", "Macro-F1": "-",
                         "C-index": "-", "Note": "no result for this cohort"})
            continue
        note = f"{p.get('source', '')}; {p.get('task', '')}"
        if p.get("comparable") is False:
            note += f"; marked not comparable in config: {p.get('why_not', '')}"
        rows.append({"Method": f"{method} (paper)", "Accuracy": fmt(p.get("accuracy")),
                     "Weighted-F1": fmt(p.get("weighted_f1")), "Macro-F1": fmt(p.get("macro_f1", p.get("f1"))),
                     "C-index": "-", "Note": note})
    for o, name in OUTS:
        r = res["results"][o]
        row = {"Method": name}
        for m, label in METRICS:
            row[label] = fmt(r[m]["mean"], r[m]["std"])
        row["C-index"] = fmt(r["c_index"]["mean"], r["c_index"]["std"]) if "c_index" in r else "-"
        row["Note"] = f"5-fold CV, {res['num_labeled']} labeled patients"
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. survival
# ---------------------------------------------------------------------------
def survival_analysis(cohort, res, pred, fig_dir):
    cfg = res["config"]
    y, ev = pred["true_label"].values, pred["os_event"].values
    strat = np.where(y >= 0, y, len(res["class_names"]) + ev)
    fold = np.zeros(len(pred), int)
    for k, (_, te) in enumerate(StratifiedKFold(cfg["n_folds"], shuffle=True, random_state=cfg["seed"])
                                .split(np.zeros(len(pred)), strat), 1):
        fold[te] = k
    d = pred.assign(fold=fold)
    n_excluded = int((d["os_time"] <= 0).sum())
    d = d[d["os_time"] > 0].copy()
    d["high"] = d.groupby("fold")["risk_score"].transform(lambda s: s > s.median())
    d["years"] = d["os_time"] / 365.25
    hi, lo = d[d["high"]], d[~d["high"]]
    lr = logrank_test(hi["years"], lo["years"], hi["os_event"], lo["os_event"])
    cph = CoxPHFitter().fit(pd.DataFrame({"t": d["years"], "e": d["os_event"], "high": d["high"].astype(int)}),
                            "t", "e").summary.loc["high"]

    fig, ax = plt.subplots(figsize=(7.5, 7))
    kms = []
    for g, lab, col in [(lo, "Low risk", "#1f77b4"), (hi, "High risk", "#d62728")]:
        km = KaplanMeierFitter(label=f"{lab} (n={len(g)}, deaths={int(g['os_event'].sum())})")
        km.fit(g["years"], g["os_event"])
        km.plot_survival_function(ax=ax, ci_show=True, color=col)
        kms.append(km)
    add_at_risk_counts(*kms, ax=ax)
    c = res["results"]["ensembled"]["c_index"]
    ax.set_title(f"TCGA-{cohort.upper()}: Ensembled Transformer-Graph Streams risk groups\n"
                 f"C-index {c['mean']:.3f} ± {c['std']:.3f} | log-rank p = {lr.p_value:.2e} | "
                 f"HR {cph['exp(coef)']:.2f}", fontsize=9)
    ax.set_xlabel("Years")
    ax.set_ylabel("Overall survival probability")
    fig.tight_layout()
    path = os.path.join(fig_dir, f"{cohort}_kaplan_meier.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)

    def s5(g):
        return float(KaplanMeierFitter().fit(g["years"], g["os_event"]).survival_function_at_times(5.0).iloc[0])
    return {"cohort": cohort.upper(), "patients": len(d), "deaths": int(d["os_event"].sum()),
            "excluded_no_survival_time": n_excluded,
            "c_index_mean": c["mean"], "c_index_std": c["std"],
            "c_index_pooled": res["pooled_out_of_fold"]["ensembled"]["c_index"],
            "deaths_high": int(hi["os_event"].sum()), "n_high": len(hi),
            "deaths_low": int(lo["os_event"].sum()), "n_low": len(lo),
            "logrank_p": float(lr.p_value), "hazard_ratio": float(cph["exp(coef)"]),
            "hr_ci_low": float(cph["exp(coef) lower 95%"]), "hr_ci_high": float(cph["exp(coef) upper 95%"]),
            "surv5_high": s5(hi), "surv5_low": s5(lo), "figure": path}


# ---------------------------------------------------------------------------
# 3. interpretation
# ---------------------------------------------------------------------------
RANKINGS = [("subtype_ensembled", "ensembled_macro_f1_drop", "ensembled_true_prob_drop", "Subtype - Ensembled Transformer-Graph Streams (macro-F1 drop)"),
            ("subtype_trans", "trans_macro_f1_drop", None, "Subtype - transformer stream alone (macro-F1 drop)"),
            ("survival_ensembled", "ensembled_c_index_drop", None, "Survival - Ensembled Transformer-Graph Streams (C-index drop)")]


def interpretation(cohort, imp, rep_dir, fig_dir):
    n_total = imp[["fold", "member"]].drop_duplicates().shape[0]
    whole = imp[imp["kind"] == "whole_omics"].groupby("omics")[
        ["ensembled_macro_f1_drop", "ensembled_true_prob_drop", "ensembled_c_index_drop", "trans_macro_f1_drop"]].agg(["mean", "std"])
    feats = imp[imp["kind"] != "whole_omics"]
    metric_cols = ["ensembled_macro_f1_drop", "ensembled_true_prob_drop", "ensembled_c_index_drop", "trans_macro_f1_drop"]
    agg = feats.groupby(["omics", "feature", "kind"])[metric_cols].sum(min_count=1).fillna(0.0) / n_total
    agg["used_by_members"] = feats.groupby(["omics", "feature", "kind"]).size()
    agg = agg.reset_index()
    agg.to_csv(os.path.join(rep_dir, f"{cohort}_feature_importance_summary.csv"), index=False)

    md = [f"Importance = performance drop when the feature is set to 0 (reference transformer paper Sec. 2.5), "
          f"averaged over all {n_total} bagging members of the 5 folds (0 for members that did not use it). "
          f"Test-fold patients only. Only features with a mean drop > 0 are listed (removing them makes the model "
          f"worse), so a list can be shorter than {TOP}. Ties in the subtype ranking are ordered by the drop in "
          f"true-class probability.\n", "### Whole-omics importance (all transformer-stream features of one omics set to 0)\n"]
    w = pd.DataFrame({"Omics": whole.index,
                      "Ensembled macro-F1 drop": [fmt(*whole.loc[o, "ensembled_macro_f1_drop"]) for o in whole.index],
                      "Ensembled C-index drop": [fmt(*whole.loc[o, "ensembled_c_index_drop"]) for o in whole.index],
                      "Transformer-stream macro-F1 drop": [fmt(*whole.loc[o, "trans_macro_f1_drop"]) for o in whole.index]})
    md.append(md_table(w) + "\n")

    fig, axes = plt.subplots(len(RANKINGS), 3, figsize=(15, 4.2 * len(RANKINGS)))
    for r, (key, col, tie, title) in enumerate(RANKINGS):
        md.append(f"### {title}: top {TOP} per omics\n")
        for c, omics in enumerate(["mrna", "mirna", "meth"]):
            # only features whose removal makes the model worse (mean drop > 0); lists can be shorter than TOP
            sub = agg[(agg["omics"] == omics) & (agg[col] > 0)]
            sub = sub.sort_values([col] + ([tie] if tie else []), ascending=False).head(TOP)
            sub.to_csv(os.path.join(rep_dir, f"{cohort}_top{TOP}_{key}_{omics}.csv"), index=False)
            with open(os.path.join(rep_dir, f"{cohort}_top{TOP}_{key}_{omics}.txt"), "w") as f:
                f.write("\n".join(sub["feature"]) + "\n")
            t = pd.DataFrame({"Rank": range(1, len(sub) + 1), "Feature": sub["feature"].values,
                              "Type": sub["kind"].values, f"Mean {col}": [f"{v:.4f}" for v in sub[col]],
                              "Used by members": [f"{u}/{n_total}" for u in sub["used_by_members"]]})
            md.append(f"**{omics}**\n\n" + md_table(t) + "\n")
            top15 = sub.head(15)[::-1]
            ax = axes[r, c]
            ax.barh(top15["feature"], top15[col], color="#4c72b0")
            ax.set_title(f"{title}\n{omics}", fontsize=8)
            ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"{cohort}_feature_importance_top15.png"), dpi=200)
    plt.close(fig)
    return "\n".join(md)


# ---------------------------------------------------------------------------
def combined_table(results):
    """One row per cohort x model (our runs) - the cross-cohort overview."""
    rows = []
    for cohort, res in results.items():
        for o, name in OUTS:
            r = res["results"][o]
            row = {"Cohort": cohort.upper(), "Model": name}
            for m, label in METRICS:
                row[label] = fmt(r[m]["mean"], r[m]["std"])
            row["C-index"] = fmt(r["c_index"]["mean"], r["c_index"]["std"]) if "c_index" in r else "-"
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    registry = [k for k in json.load(open("configs/cohorts.json")) if not k.startswith("_")]
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="all", choices=registry + ["all"])
    ap.add_argument("--tag", default="final")
    ap.add_argument("--results_dir", default=None, help="default results/<tag>")
    ap.add_argument("--out_dir", default=None, help="default results/<tag>/report (figures: results/figures/<tag>)")
    a = ap.parse_args()
    res_dir = a.results_dir or os.path.join("results", a.tag)
    rep_dir = a.out_dir or os.path.join(res_dir, "report")
    # figures go next to the report when a custom folder is given, else to results/figures/<tag>
    fig_dir = (os.path.join(rep_dir, "figures") if (a.out_dir or a.results_dir)
               else os.path.join("results", "figures", a.tag))
    os.makedirs(rep_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    cohorts = registry if a.cohort == "all" else [a.cohort]
    loaded = {c: json.load(open(os.path.join(res_dir, f"{c}_5fold_results.json")))
              for c in cohorts if os.path.exists(os.path.join(res_dir, f"{c}_5fold_results.json"))}
    md, surv_rows = [f"# Results report ({a.tag})\n", "Generated by src/report_exact.py from saved result files.\n"], []
    if loaded:
        ct_all = combined_table(loaded)
        ct_all.to_csv(os.path.join(rep_dir, "all_cohorts_our_models.csv"), index=False)
        md.append("## All cohorts - our models (5-fold mean ± std)\n\n" + md_table(ct_all) + "\n")
    for cohort in cohorts:
        rp = os.path.join(res_dir, f"{cohort}_5fold_results.json")
        if not os.path.exists(rp):
            print(f"[{cohort}] no results in {res_dir} - skipped")
            continue
        res = json.load(open(rp))
        pred = pd.read_csv(os.path.join(res_dir, f"{cohort}_predictions.csv"))
        md.append(f"## TCGA-{cohort.upper()} ({res['num_labeled']} labeled of {res['num_samples']} patients)\n")

        ct = classification_table(cohort, res)
        ct.to_csv(os.path.join(rep_dir, f"{cohort}_classification.csv"), index=False)
        md.append("### 1. Subtype classification\n\n" + md_table(ct) + "\n")

        s = survival_analysis(cohort, res, pred, fig_dir)
        surv_rows.append(s)
        md.append("### 2. Survival prediction (Ensembled Transformer-Graph Streams, out-of-fold risk scores)\n\n" + md_table(pd.DataFrame([{
            "Patients / deaths": f"{s['patients']} / {s['deaths']}",
            "C-index (5-fold)": fmt(s["c_index_mean"], s["c_index_std"]),
            "Deaths high vs low risk": f"{s['deaths_high']}/{s['n_high']} vs {s['deaths_low']}/{s['n_low']}",
            "Log-rank p": f"{s['logrank_p']:.2e}",
            "Hazard ratio (95% CI)": f"{s['hazard_ratio']:.2f} ({s['hr_ci_low']:.2f}-{s['hr_ci_high']:.2f})",
            "5-year survival high vs low": f"{s['surv5_high']:.1%} vs {s['surv5_low']:.1%}"}])) +
            f"\n\nRisk groups: split at the median risk of each test fold. "
            f"{s['excluded_no_survival_time']} patients without survival time excluded here. "
            f"Figure: `{s['figure']}`\n")

        ip = os.path.join(res_dir, f"{cohort}_feature_importance.csv")
        md.append("### 3. Biological interpretation\n")
        if os.path.exists(ip):
            md.append(interpretation(cohort, pd.read_csv(ip), rep_dir, fig_dir))
        else:
            md.append("Not available: this cohort was trained without `--interpret`.\n")
        print(f"[{cohort}] done")

    if surv_rows:
        pd.DataFrame(surv_rows).to_csv(os.path.join(rep_dir, "survival_summary.csv"), index=False)
    out = os.path.join(rep_dir, "REPORT.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"Report written to {out}; figures in {fig_dir}")


if __name__ == "__main__":
    main()
