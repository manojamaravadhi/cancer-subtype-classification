"""
Publication figures for the exact-stream pipeline. Reads saved files only (results of
src/train_exact.py, tables of src/report_exact.py, enrichment of src/enrichment_exact.py,
processed data for the heatmap) - nothing is trained or re-scored.

Main figures                                   Supplementary figures
  fig1 classification vs base papers             S1 learning / loss curves (transformer stream, graph stream, fusion)
  fig2 confusion matrices (Ensembled Transformer-Graph Streams)           S2 per-fold scores + bagging benefit
  fig3 Kaplan-Meier risk groups + HR forest      S3 per-class recall of the three models
  fig4 top biomarkers (subtype, survival)        S4 ROC curves (Ensembled Transformer-Graph Streams, one-vs-rest)
       + whole-omics importance                  S5 risk score per subtype
  fig5 heatmap of top biomarkers by subtype      S6 driver vs non-driver biomarkers, stability
  fig6 GO / KEGG enrichment

Colours: the validated reference categorical palette (slots 1-3 for our three models,
fixed order, same everywhere); risk groups use the diverging blue / red poles; paper
values are dark markers distinguished by shape. Each figure is saved as PNG (300 dpi)
and PDF in results/figures/<tag>/{main,supplementary}/.

Usage: python -m src.plot_exact [--tag final] [--results_dir DIR --out_dir DIR]
"""

import os
import json
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, recall_score, roc_curve, auc
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.plotting import add_at_risk_counts

# ---- palette (reference instance, light mode) -------------------------------------
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MODEL_COLOR = {"trans": SERIES[0], "graph": SERIES[1], "ensembled": SERIES[2]}
MODEL_NAME = {"trans": "Transformer stream", "graph": "Graph stream", "ensembled": "Ensembled Transformer-Graph Streams (ours)"}
LOW, HIGH = "#2a78d6", "#e34948"                         # diverging poles (low / high risk)
BLUES = LinearSegmentedColormap.from_list("blues", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DIVERGE = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#86b6ef", "#f0efec", "#f2a3a2", "#b8302f"])
PAPER_MARK = {"Reference transformer method": "D", "Reference graph method": "s", "FastMI-HGNet": "^"}
KIND_COLOR = {"driver": SERIES[0], "l1svc": SERIES[1]}
KIND_NAME = {"driver": "cancer driver (NCG / OncomiR)", "l1svc": "L1-SVC selected"}
OMICS = [("mrna", "mRNA"), ("mirna", "miRNA"), ("meth", "Methylation")]
METRICS = [("accuracy", "Accuracy"), ("weighted_f1", "Weighted-F1"), ("macro_f1", "Macro-F1")]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "lines.linewidth": 2, "legend.frameon": False, "pdf.fonttype": 42})


def save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    for ext, kw in (("png", {"dpi": 300}), ("pdf", {})):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), bbox_inches="tight", **kw)
    plt.close(fig)
    print(f"  {name}")


def note(ax, text):
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes, color=INK2, fontsize=8, wrap=True)
    ax.set_axis_off()


def folds_of(res, pred):
    cfg = res["config"]
    y, ev = pred["true_label"].values, pred["os_event"].values
    strat = np.where(y >= 0, y, len(res["class_names"]) + ev)
    fold = np.zeros(len(pred), int)
    for k, (_, te) in enumerate(StratifiedKFold(cfg["n_folds"], shuffle=True, random_state=cfg["seed"])
                                .split(np.zeros(len(pred)), strat), 1):
        fold[te] = k
    return fold


# =================================================================================
# Main figures
# =================================================================================
def fig1_classification(D, out):
    papers = json.load(open("configs/base_paper_results.json"))
    cohorts = list(D)
    fig, axes = plt.subplots(1, len(cohorts), figsize=(4.6 * len(cohorts), 4.2), sharey=True, squeeze=False)
    handles = {}
    for ax, c in zip(axes[0], cohorts):
        res = D[c]["res"]
        x = np.arange(len(METRICS))
        w = 0.24
        for j, o in enumerate(["trans", "graph", "ensembled"]):
            m = [res["results"][o][k]["mean"] for k, _ in METRICS]
            s = [res["results"][o][k]["std"] for k, _ in METRICS]
            b = ax.bar(x + (j - 1) * w, m, w * 0.92, yerr=s, color=MODEL_COLOR[o], label=MODEL_NAME[o],
                       error_kw={"ecolor": INK2, "elinewidth": 1, "capsize": 2})
            handles[MODEL_NAME[o]] = b
        for name, mk in PAPER_MARK.items():
            key = json.load(open("configs/cohorts.json")).get(c, {}).get("paper_key", c)   # e.g. brca_pancan -> brca
            p = (papers.get(name) or {}).get(key)
            if not p:
                continue
            filled = p.get("comparable", True) is not False
            for i, (k, _) in enumerate(METRICS):
                if p.get(k) is not None:
                    h = ax.scatter(i + 0.42, p[k], marker=mk, s=42, zorder=5, edgecolor=INK, linewidth=1.1,
                                   facecolor=INK if filled else "none")
                    handles[f"{name} paper" + ("" if filled else " (not comparable, see config)")] = h
        ax.set_xticks(x)
        ax.set_xticklabels([l for _, l in METRICS])
        ax.set_ylim(0, 1.05)
        ax.set_title(f"TCGA-{c.upper()} ({res['num_labeled']} labeled)")
    axes[0][0].set_ylabel("Score (5-fold mean ± SD)")
    fig.legend(handles.values(), handles.keys(), loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.12))
    save(fig, out, "fig1_classification_vs_papers")


def fig2_confusion(D, out):
    cohorts = list(D)
    fig, axes = plt.subplots(1, len(cohorts), figsize=(4.3 * len(cohorts), 4), squeeze=False)
    for ax, c in zip(axes[0], cohorts):
        res, pred = D[c]["res"], D[c]["pred"]
        lab = pred[pred["true_label"] >= 0]
        names = res["class_names"]
        cm = confusion_matrix(lab["true_label"], lab["pred_ensembled"], labels=range(len(names)))
        pct = cm / cm.sum(1, keepdims=True)
        ax.imshow(pct, cmap=BLUES, vmin=0, vmax=1)
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, f"{cm[i, j]}\n{pct[i, j]:.0%}", ha="center", va="center", fontsize=7,
                        color="#ffffff" if pct[i, j] > 0.55 else INK)
        short = [n.replace("LGG_", "") for n in names]
        ax.set_xticks(range(len(names)), short, rotation=30, ha="right")
        ax.set_yticks(range(len(names)), short)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.grid(False)
        ax.set_title(f"TCGA-{c.upper()}")
    fig.suptitle("Ensembled Transformer-Graph Streams - out-of-fold confusion matrices (count, row %)", y=1.02)
    fig.tight_layout()
    save(fig, out, "fig2_confusion_ensembled")


def risk_groups(res, pred):
    d = pred.assign(fold=folds_of(res, pred))
    d = d[d["os_time"] > 0].copy()
    d["high"] = d.groupby("fold")["risk_score"].transform(lambda s: s > s.median())
    d["years"] = d["os_time"] / 365.25
    return d


def fig3_survival(D, out):
    cohorts = list(D)
    fig, axes = plt.subplots(1, len(cohorts), figsize=(5.2 * len(cohorts), 5.6), squeeze=False)
    forest = []
    for ax, c in zip(axes[0], cohorts):
        res = D[c]["res"]
        d = risk_groups(res, D[c]["pred"])
        kms = []
        for flag, name, col in [(False, "Low risk", LOW), (True, "High risk", HIGH)]:
            g = d[d["high"] == flag]
            km = KaplanMeierFitter(label=name).fit(g["years"], g["os_event"])
            km.plot_survival_function(ax=ax, ci_show=True, color=col, ci_alpha=0.15)
            kms.append(km)
        add_at_risk_counts(*kms, ax=ax, rows_to_show=["At risk"])
        cph = CoxPHFitter().fit(pd.DataFrame({"t": d["years"], "e": d["os_event"],
                                              "high": d["high"].astype(int)}), "t", "e").summary.loc["high"]
        from lifelines.statistics import logrank_test
        hi, lo = d[d["high"]], d[~d["high"]]
        p = logrank_test(hi["years"], lo["years"], hi["os_event"], lo["os_event"]).p_value
        ci = res["results"]["ensembled"]["c_index"]
        ax.set_title(f"TCGA-{c.upper()}\nC-index {ci['mean']:.3f} ± {ci['std']:.3f} | log-rank p = {p:.1e}")
        ax.set_xlabel("Years")
        ax.set_ylabel("Overall survival")
        ax.set_ylim(0, 1.02)
        forest.append((c.upper(), cph["exp(coef)"], cph["exp(coef) lower 95%"], cph["exp(coef) upper 95%"], p))
    fig.tight_layout()
    save(fig, out, "fig3a_kaplan_meier")

    fig, ax = plt.subplots(figsize=(5.2, 0.7 + 0.55 * len(forest)))
    for i, (c, hr, lo_, hi_, p) in enumerate(forest[::-1]):
        ax.plot([lo_, hi_], [i, i], color=INK2, lw=1.5)
        ax.scatter(hr, i, s=60, color=SERIES[2], zorder=3, edgecolor=SURFACE, linewidth=2)
        ax.text(hi_ * 1.08, i, f"HR {hr:.2f} ({lo_:.2f}-{hi_:.2f}), p = {p:.1e}", va="center", fontsize=8, color=INK)
    ax.axvline(1, color=INK2, lw=1, ls="--")
    ax.set_xscale("log")
    ax.set_yticks(range(len(forest)), [f[0] for f in forest[::-1]])
    ax.set_xlabel("Hazard ratio, high vs low predicted risk (log scale, 95% CI)")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0.8, max(f[3] for f in forest) * 4)
    save(fig, out, "fig3b_hazard_ratio_forest")


def load_top(rep, c, key, omics):
    p = os.path.join(rep, f"{c}_top30_{key}_{omics}.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


def fig4_biomarkers(D, rep, out):
    for c in D:
        if not D[c]["has_imp"]:
            continue
        n_total = D[c]["imp"][["fold", "member"]].drop_duplicates().shape[0]
        rows = [("subtype_ensembled", "ensembled_macro_f1_drop", "Subtype: macro-F1 drop"),
                ("survival_ensembled", "ensembled_c_index_drop", "Survival: C-index drop")]
        fig, axes = plt.subplots(2, 3, figsize=(13, 9.5))
        for r, (key, col, label) in enumerate(rows):
            for k, (o, oname) in enumerate(OMICS):
                ax = axes[r, k]
                t = load_top(rep, c, key, o)
                if t is None or t.empty:
                    note(ax, f"{oname}: no feature with a positive drop")
                    continue
                t = t.head(15)[::-1]
                # drawn in units of 10^-3 so the axis needs no separate offset label
                ax.barh(t["feature"], t[col] * 1e3, color=[KIND_COLOR[x] for x in t["kind"]], height=0.72)
                ax.set_title(f"{oname} - {label}", fontsize=9)
                ax.tick_params(axis="y", labelsize=7)
                ax.grid(axis="y", visible=False)
                ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
                ax.set_xlabel(f"Mean drop (×10⁻³) over {n_total} bagging members (test folds)", fontsize=8)
        handles = [plt.Rectangle((0, 0), 1, 1, color=v) for v in KIND_COLOR.values()]
        fig.legend(handles, KIND_NAME.values(), loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
        fig.suptitle(f"TCGA-{c.upper()} - top biomarkers (feature set to 0, reference transformer paper Sec. 2.5)", y=1.0)
        fig.tight_layout(rect=(0, 0.03, 1, 0.98))
        save(fig, out, f"fig4_{c}_top_biomarkers")

    cs = [c for c in D if D[c]["has_imp"]]
    if not cs:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, (col, label) in zip(axes, [("ensembled_macro_f1_drop", "Subtype: macro-F1 drop"),
                                       ("ensembled_c_index_drop", "Survival: C-index drop")]):
        w = 0.8 / len(cs)
        for j, c in enumerate(cs):
            imp = D[c]["imp"]
            wo = imp[imp["kind"] == "whole_omics"].groupby("omics")[col].agg(["mean", "std"])
            vals = [wo.loc[o, "mean"] if o in wo.index else np.nan for o, _ in OMICS]
            sds = [wo.loc[o, "std"] if o in wo.index else np.nan for o, _ in OMICS]
            ax.bar(np.arange(3) + (j - (len(cs) - 1) / 2) * w, vals, w * 0.92, yerr=sds, label=c.upper(),
                   color=SERIES[3 + j % 5], error_kw={"ecolor": INK2, "elinewidth": 1, "capsize": 2})
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xticks(range(3), [n for _, n in OMICS])
        ax.set_title(f"Whole-omics importance - {label}")
        ax.set_ylabel("Mean drop ± SD over members")
    axes[0].legend()
    fig.tight_layout()
    save(fig, out, "fig4_whole_omics_importance")


def fig5_heatmaps(D, rep, out):
    for c in D:
        if not D[c]["has_imp"]:
            continue
        res = D[c]["res"]
        d = f"data/processed_exact/tcga_{c}"
        y = np.load(f"{d}/y_subtype.npy")
        names = json.load(open(f"{d}/feature_names.json"))
        blocks, labels = [], []
        for o, oname in OMICS:
            t = load_top(rep, c, "subtype_ensembled", o)
            if t is None or t.empty:
                continue
            X = np.load(f"{d}/X_trans_{o}.npy")
            idx = {n: i for i, n in enumerate(names[f"trans_{o}"])}
            for f in t["feature"].head(10):
                v = X[:, idx[f]].astype(float)
                blocks.append((v - v[y >= 0].mean()) / (v[y >= 0].std() + 1e-8))
                labels.append(f"{f} ({oname})")
        if not blocks:
            continue
        M = np.array(blocks)
        order = np.concatenate([np.where(y == k)[0] for k in range(len(res["class_names"]))])
        fig = plt.figure(figsize=(11, 0.22 * len(labels) + 2.2))
        gs = fig.add_gridspec(2, 2, height_ratios=[0.35, len(labels) * 0.22], width_ratios=[40, 1], hspace=0.04,
                              top=0.86, bottom=0.08)
        ax0, ax = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])
        cls_cols = [SERIES[k % 8] for k in range(len(res["class_names"]))]
        ax0.imshow([[k for k in y[order]]], aspect="auto",
                   cmap=matplotlib.colors.ListedColormap(cls_cols), vmin=-0.5, vmax=len(cls_cols) - 0.5)
        ax0.set_axis_off()
        im = ax.imshow(np.clip(M[:, order], -2.5, 2.5), aspect="auto", cmap=DIVERGE, vmin=-2.5, vmax=2.5,
                       interpolation="nearest")
        ax.set_yticks(range(len(labels)), labels, fontsize=7)
        ax.set_xticks([])
        ax.set_xlabel(f"Labeled patients ({len(order)}), grouped by subtype")
        ax.grid(False)
        cax = fig.add_subplot(gs[1, 1])
        fig.colorbar(im, cax=cax, label="z-score (clipped ±2.5)")
        handles = [plt.Rectangle((0, 0), 1, 1, color=col) for col in cls_cols]
        fig.legend(handles, [n.replace("LGG_", "") for n in res["class_names"]], loc="lower center",
                   ncol=len(cls_cols), bbox_to_anchor=(0.45, 0.875))
        fig.suptitle(f"TCGA-{c.upper()} - top-10 subtype biomarkers per omics (Ensembled Transformer-Graph Streams)", y=0.95)
        save(fig, out, f"fig5_{c}_biomarker_heatmap")


def fig6_enrichment(D, rep, out):
    enr = os.path.join(rep, "enrichment")
    for c in D:
        lists = [(f"{c}_{r}_{o}", t) for r, rt in [("subtype_ensembled", "Subtype"), ("survival_ensembled", "Survival")]
                 for o, t in [("mrna", f"{rt} - mRNA genes"), ("meth", f"{rt} - methylation genes")]]
        lists = [(k, t) for k, t in lists if os.path.exists(os.path.join(enr, f"{k}.csv"))]
        if not lists:
            continue
        fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))
        for ax, (k, title) in zip(axes.flat, lists):
            r = pd.read_csv(os.path.join(enr, f"{k}.csv"))
            r = r[r["Adjusted P-value"] < 0.05].sort_values("Adjusted P-value").head(10)
            if r.empty:
                note(ax, f"{title}\nno GO / KEGG term with adjusted p < 0.05")
                continue
            r = r[::-1]
            n_hit = r["Overlap"].str.split("/").str[0].astype(int)
            col = [SERIES[0] if s.startswith("GO") else SERIES[1] for s in r["Gene_set"]]
            ax.scatter(-np.log10(r["Adjusted P-value"]), range(len(r)), s=30 + 25 * n_hit, c=col,
                       edgecolor=SURFACE, linewidth=1.5, zorder=3)
            ax.set_yticks(range(len(r)), [t if len(t) < 60 else t[:57] + "..." for t in r["Term"]], fontsize=7)
            ax.set_xlabel("-log10 adjusted p (Benjamini-Hochberg)")
            ax.axvline(-np.log10(0.05), color=INK2, lw=0.8, ls="--")
            ax.set_title(title, fontsize=9)
            ax.grid(axis="y", visible=False)
        for ax in list(axes.flat)[len(lists):]:
            ax.set_axis_off()
        handles = [plt.Line2D([], [], marker="o", ls="", color=SERIES[0], markersize=7),
                   plt.Line2D([], [], marker="o", ls="", color=SERIES[1], markersize=7)]
        fig.legend(handles, ["GO Biological Process", "KEGG pathway"], loc="lower center", ncol=2,
                   bbox_to_anchor=(0.5, -0.01))
        fig.suptitle(f"TCGA-{c.upper()} - enrichment of top biomarker genes (Enrichr; dot size = genes in term)")
        fig.tight_layout(rect=(0, 0.03, 1, 0.97))
        save(fig, out, f"fig6_{c}_enrichment")


# =================================================================================
# Supplementary figures
# =================================================================================
def stack(curves, getter):
    seqs = [np.asarray([np.nan if v is None else v for v in getter(m)], float) for m in curves.values()
            if getter(m) is not None]
    seqs = [s for s in seqs if len(s)]
    if not seqs:
        return None
    L = max(len(s) for s in seqs)
    A = np.full((len(seqs), L), np.nan)
    for i, s in enumerate(seqs):
        A[i, :len(s)] = s
    return A


def draw_curves(ax, A, color, label, min_members=5):
    """Thin line per member + mean while at least `min_members` members are still training."""
    if A is None:
        return
    x = np.arange(1, A.shape[1] + 1)
    for row in A:
        ax.plot(x, row, color=color, lw=0.5, alpha=0.18)
    n = (~np.isnan(A)).sum(0)
    with np.errstate(all="ignore"):
        mean = np.where(n >= min_members, np.nanmean(A, 0), np.nan)
    ax.plot(x, mean, color=color, lw=2, label=label)


def s1_learning_curves(D, out):
    for c in D:
        cv = D[c]["curves"]
        fig, axes = plt.subplots(2, 3, figsize=(14, 7.6))
        ax = axes[0, 0]
        draw_curves(ax, stack(cv, lambda m: m["trans"]["loss"]), SERIES[0], "training loss")
        pre = D[c]["res"]["config"]["trans_pretrain_epochs"]
        ax.axvspan(0.5, pre + 0.5, color=GRID, alpha=0.6, lw=0)
        ax.text(pre / 2 + 0.5, ax.get_ylim()[1] * 0.97, "pre-train", ha="center", va="top", fontsize=7, color=INK2)
        ax.set_title("Transformer stream - training loss")
        ax = axes[0, 1]
        draw_curves(ax, stack(cv, lambda m: m["trans"]["oob_acc"]), SERIES[0], "OOB accuracy")
        draw_curves(ax, stack(cv, lambda m: m["trans"]["oob_macro_f1"]), SERIES[1], "OOB macro-F1")
        ax.set_ylim(0, 1)
        ax.set_title("Transformer stream - out-of-bag scores (epoch selection)")
        ax.legend(loc="lower right")
        ax = axes[0, 2]
        if all("graph" in m for m in cv.values()):
            for i, (o, oname) in enumerate(OMICS):
                draw_curves(ax, stack(cv, lambda m, o=o: m["graph"]["gcn_loss"][o]), SERIES[i], f"GCN {oname}")
            draw_curves(ax, stack(cv, lambda m: m["graph"]["gat_loss"]), SERIES[6], "GAT (fused graph)")
            ax.set_title("Graph stream - training loss")
            ax.legend(loc="upper right")
        else:
            note(ax, "Graph-stream loss values were not saved in this run")
        ax = axes[1, 0]
        draw_curves(ax, stack(cv, lambda m: m["fusion"]["loss"]), SERIES[2], "training loss")
        ax.set_title("Fusion - training loss (focal + 0.15 × Cox)")
        ax = axes[1, 1]
        draw_curves(ax, stack(cv, lambda m: m["fusion"]["oob_macro_f1"]), SERIES[2], "OOB macro-F1")
        draw_curves(ax, stack(cv, lambda m: m["fusion"]["oob_c_index"]), SERIES[7], "OOB C-index")
        ax.set_ylim(0, 1)
        ax.set_title("Fusion - out-of-bag scores (epoch selection)")
        ax.legend(loc="lower right")
        ax = axes[1, 2]
        eps = [mm["trans_best_epoch"] for f in D[c]["res"]["folds"] for mm in f["members"]]
        feps = [mm["fusion_best_epoch"] for f in D[c]["res"]["folds"] for mm in f["members"]]
        ax.hist([eps, feps], bins=15, color=[SERIES[0], SERIES[2]],
                label=["Transformer stream (epoch after pre-training)", "Fusion"])
        ax.set_title("Chosen epoch per member (out-of-bag)")
        ax.set_xlabel("Chosen epoch")
        ax.set_ylabel("Members")
        ax.legend()
        for ax in list(axes.flat)[:5]:
            ax.set_xlabel("Epoch")
        for ax in axes[0, :2]:
            ax.set_xlabel(f"Epoch (first {pre} = pre-training)")
        fig.suptitle(f"TCGA-{c.upper()} - learning curves of all {len(cv)} bagging members "
                     f"(thin: members; thick: mean while ≥5 members train)")
        fig.tight_layout()
        save(fig, out, f"S1_{c}_learning_curves")


def s2_folds_bagging(D, out):
    cohorts = list(D)
    fig, axes = plt.subplots(1, len(cohorts), figsize=(4.4 * len(cohorts), 4), sharey=True, squeeze=False)
    rng = np.random.default_rng(0)                       # horizontal jitter of dots only
    for ax, c in zip(axes[0], cohorts):
        res = D[c]["res"]
        groups = [("Transformer stream\n(bagged)", [f["trans"]["macro_f1"] for f in res["folds"]], SERIES[0]),
                  ("Graph stream\n(bagged)", [f["graph"]["macro_f1"] for f in res["folds"]], SERIES[1]),
                  ("Ensembled streams\nsingle member", [m["ensembled_test_member_alone"]["macro_f1"] for f in res["folds"]
                                           for m in f["members"]], "#86b6ef"),
                  ("Ensembled streams\n(bagged)", [f["ensembled"]["macro_f1"] for f in res["folds"]], SERIES[2])]
        for i, (name, vals, col) in enumerate(groups):
            ax.scatter(i + rng.uniform(-0.12, 0.12, len(vals)), vals, s=34, color=col, edgecolor=SURFACE,
                       linewidth=1.5, zorder=3)
            ax.plot([i - 0.25, i + 0.25], [np.mean(vals)] * 2, color=INK, lw=2)
        ax.set_xticks(range(len(groups)), [g[0] for g in groups], fontsize=8)
        ax.set_title(f"TCGA-{c.upper()}")
        ax.set_ylim(0, 1.02)
    axes[0][0].set_ylabel("Test-fold macro-F1 (bar = mean)")
    fig.suptitle("Per-fold scores and the effect of bagging (single members vs weighted ensemble)", y=1.02)
    fig.tight_layout()
    save(fig, out, "S2_per_fold_and_bagging")


def s3_per_class(D, out):
    cohorts = list(D)
    fig, axes = plt.subplots(1, len(cohorts), figsize=(4.6 * len(cohorts), 4), sharey=True, squeeze=False)
    for ax, c in zip(axes[0], cohorts):
        res, pred = D[c]["res"], D[c]["pred"]
        lab = pred[pred["true_label"] >= 0]
        names = res["class_names"]
        n = np.bincount(lab["true_label"], minlength=len(names))
        x, w = np.arange(len(names)), 0.26
        for j, o in enumerate(["trans", "graph", "ensembled"]):
            r = recall_score(lab["true_label"], lab[f"pred_{o}"], labels=range(len(names)), average=None, zero_division=0)
            ax.bar(x + (j - 1) * w, r, w * 0.92, color=MODEL_COLOR[o], label=MODEL_NAME[o])
        ax.set_xticks(x, [f"{nm.replace('LGG_', '')}\n(n={k})" for nm, k in zip(names, n)], fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"TCGA-{c.upper()}")
    axes[0][0].set_ylabel("Recall (out-of-fold)")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, out, "S3_per_class_recall")


def s4_roc(D, out):
    cohorts = list(D)
    fig, axes = plt.subplots(1, len(cohorts), figsize=(4.4 * len(cohorts), 4.2), squeeze=False)
    for ax, c in zip(axes[0], cohorts):
        res, pred = D[c]["res"], D[c]["pred"]
        lab = pred[pred["true_label"] >= 0]
        for k, nm in enumerate(res["class_names"]):
            fpr, tpr, _ = roc_curve((lab["true_label"] == k).astype(int), lab[f"prob_{nm}"])
            ax.plot(fpr, tpr, color=SERIES[k % 8], label=f"{nm.replace('LGG_', '')} (AUC {auc(fpr, tpr):.3f})")
        ax.plot([0, 1], [0, 1], color=INK2, lw=0.8, ls="--")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title(f"TCGA-{c.upper()} - Ensembled Transformer-Graph Streams")
        ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    save(fig, out, "S4_roc_ensembled")


def s5_risk_by_subtype(D, out):
    cohorts = list(D)
    fig, axes = plt.subplots(1, len(cohorts), figsize=(4.4 * len(cohorts), 4), squeeze=False)
    for ax, c in zip(axes[0], cohorts):
        res, pred = D[c]["res"], D[c]["pred"]
        names = res["class_names"]
        data = [pred.loc[pred["true_label"] == k, "risk_score"].dropna().values for k in range(len(names))]
        parts = ax.violinplot(data, showmedians=True, showextrema=False)
        for k, b in enumerate(parts["bodies"]):
            b.set_facecolor(SERIES[k % 8])
            b.set_alpha(0.55)
        parts["cmedians"].set_color(INK)
        ev = [int(pred.loc[pred["true_label"] == k, "os_event"].sum()) for k in range(len(names))]
        ax.set_xticks(range(1, len(names) + 1),
                      [f"{nm.replace('LGG_', '')}\nn={len(d)}\ndeaths={e}" for nm, d, e in zip(names, data, ev)],
                      fontsize=7)
        ax.set_title(f"TCGA-{c.upper()}")
        ax.set_ylabel("Predicted risk (out-of-fold, standardised)")
    fig.tight_layout()
    save(fig, out, "S5_risk_by_subtype")


def s6_driver_stability(D, rep, out):
    cs = [c for c in D if D[c]["has_imp"]]
    if not cs:
        return
    fig, axes = plt.subplots(2, len(cs), figsize=(4.4 * len(cs), 7), squeeze=False)
    for j, c in enumerate(cs):
        n_total = D[c]["imp"][["fold", "member"]].drop_duplicates().shape[0]
        ax = axes[0, j]
        counts = {k: [] for k in KIND_COLOR}
        stab = []
        for o, oname in OMICS:
            t = load_top(rep, c, "subtype_ensembled", o)
            t = t if t is not None else pd.DataFrame(columns=["kind", "used_by_members"])
            for k in KIND_COLOR:
                counts[k].append(int((t["kind"] == k).sum()))
            stab.append(t["used_by_members"].values / n_total)
        bottom = np.zeros(3)
        for k, v in counts.items():
            ax.bar(range(3), v, 0.6, bottom=bottom, color=KIND_COLOR[k], label=KIND_NAME[k], edgecolor=SURFACE, lw=2)
            bottom += np.array(v)
        ax.set_xticks(range(3), [n for _, n in OMICS])
        ax.set_ylabel("Top subtype biomarkers")
        ax.set_title(f"TCGA-{c.upper()} - driver vs selected")
        ax = axes[1, j]
        for i, s in enumerate(stab):
            ax.scatter(np.full(len(s), i) + np.linspace(-0.15, 0.15, len(s)) if len(s) else [], s, s=24,
                       color=SERIES[2], edgecolor=SURFACE, linewidth=0.5)
        ax.set_xticks(range(3), [n for _, n in OMICS])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(f"Share of the {n_total} members using it")
        ax.set_title("Stability of the top biomarkers")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, out, "S6_driver_share_and_stability")


# =================================================================================
def main():
    registry = [k for k in json.load(open("configs/cohorts.json")) if not k.startswith("_")]
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="final")
    ap.add_argument("--results_dir", default=None, help="default results/<tag>")
    ap.add_argument("--report_dir", default=None, help="default <results_dir>/report")
    ap.add_argument("--out_dir", default=None, help="default results/figures/<tag>")
    a = ap.parse_args()
    res_dir = a.results_dir or os.path.join("results", a.tag)
    rep = a.report_dir or os.path.join(res_dir, "report")
    out = a.out_dir or os.path.join("results", "figures", a.tag)
    D = {}
    for c in registry:
        p = os.path.join(res_dir, f"{c}_5fold_results.json")
        if not os.path.exists(p):
            continue
        ip = os.path.join(res_dir, f"{c}_feature_importance.csv")
        D[c] = {"res": json.load(open(p)), "pred": pd.read_csv(os.path.join(res_dir, f"{c}_predictions.csv")),
                "curves": json.load(open(os.path.join(res_dir, f"{c}_learning_curves.json"))),
                "has_imp": os.path.exists(ip), "imp": pd.read_csv(ip) if os.path.exists(ip) else None}
    if not D:
        print(f"no results in {res_dir}")
        return
    main_out, sup_out = os.path.join(out, "main"), os.path.join(out, "supplementary")
    print(f"cohorts: {list(D)}\nmain figures -> {main_out}")
    fig1_classification(D, main_out)
    fig2_confusion(D, main_out)
    fig3_survival(D, main_out)
    fig4_biomarkers(D, rep, main_out)
    fig5_heatmaps(D, rep, main_out)
    fig6_enrichment(D, rep, main_out)
    print(f"supplementary figures -> {sup_out}")
    s1_learning_curves(D, sup_out)
    s2_folds_bagging(D, sup_out)
    s3_per_class(D, sup_out)
    s4_roc(D, sup_out)
    s5_risk_by_subtype(D, sup_out)
    s6_driver_stability(D, rep, sup_out)


if __name__ == "__main__":
    main()
