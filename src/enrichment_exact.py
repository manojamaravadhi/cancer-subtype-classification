"""
GO / KEGG enrichment of the biomarker lists made by src/report_exact.py, as in the
reference transformer paper (Sec. 2.5: GO enrichment and KEGG pathway analysis of the top features,
FDR < 0.05).

Input : results/<tag>/report/<cohort>_top30_<ranking>_<omics>.txt
        rankings subtype_ensembled, survival_ensembled; omics mrna and meth (both are gene symbols;
        methylation is gene-level, reference transformer paper Sec. 4.2). miRNA lists are not gene sets and are
        not enriched here.
Method: Enrichr (via gseapy) - over-representation test, Benjamini-Hochberg adjusted p
        ("Adjusted P-value"); Enrichr's default background. Needs internet access.
Output: results/<tag>/report/enrichment/<cohort>_<ranking>_<omics>.csv (all terms returned)
        + enrichment_summary.csv (terms with adjusted p < 0.05) + enrichment_info.json
        (library names, gseapy version, date, gene lists) for the methods section.

Usage:  python -m src.enrichment_exact [--tag final] [--cohort all]
"""

import os
import json
import time
import argparse

import pandas as pd
import gseapy

LIBRARIES = ["GO_Biological_Process_2025", "KEGG_2021_Human"]
RANKINGS = ["subtype_ensembled", "survival_ensembled"]
OMICS = ["mrna", "meth"]
FDR = 0.05


def main():
    registry = [k for k in json.load(open("configs/cohorts.json")) if not k.startswith("_")]
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="final")
    ap.add_argument("--cohort", default="all", choices=registry + ["all"])
    a = ap.parse_args()
    rep = os.path.join("results", a.tag, "report")
    out = os.path.join(rep, "enrichment")
    os.makedirs(out, exist_ok=True)
    info = {"libraries": LIBRARIES, "fdr_cutoff": FDR, "gseapy_version": gseapy.__version__,
            "date": time.strftime("%Y-%m-%d %H:%M:%S"), "service": "Enrichr (maayanlab.cloud/Enrichr)",
            "background": "Enrichr default", "gene_lists": {}}
    sig = []
    for c in (registry if a.cohort == "all" else [a.cohort]):
        for r in RANKINGS:
            for o in OMICS:
                path = os.path.join(rep, f"{c}_top30_{r}_{o}.txt")
                if not os.path.exists(path):
                    continue
                genes = [g.strip() for g in open(path) if g.strip()]
                key = f"{c}_{r}_{o}"
                info["gene_lists"][key] = genes
                if len(genes) < 3:
                    print(f"[{key}] only {len(genes)} genes - skipped")
                    continue
                res = gseapy.enrichr(gene_list=genes, gene_sets=LIBRARIES, organism="human",
                                     outdir=None, cutoff=1.0).results
                res.insert(0, "list", key)
                res.to_csv(os.path.join(out, f"{key}.csv"), index=False)
                s = res[res["Adjusted P-value"] < FDR]
                sig.append(s)
                print(f"[{key}] {len(genes)} genes -> {len(res)} terms tested, {len(s)} with adjusted p < {FDR}")
                time.sleep(1.0)          # be gentle with the public Enrichr service
    summary = pd.concat(sig) if sig else pd.DataFrame()
    summary.to_csv(os.path.join(out, "enrichment_summary.csv"), index=False)
    with open(os.path.join(out, "enrichment_info.json"), "w") as f:
        json.dump(info, f, indent=1)
    print(f"written to {out}")


if __name__ == "__main__":
    main()
