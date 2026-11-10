"""
Cancer-type-specific driver lists for the transformer stream (Stream A).

reference transformer method keeps "all driver genes" but its released data uses a small, cancer-specific
NCG subset (e.g. 79 mRNA drivers for BRCA, 75 of them canonical). The paper does
not state the rule, so we rebuild it reproducibly from the NCG 7 paper tables:

  gene drivers  = canonical cancer drivers (Table S3) reported in >= 1 cancer
                  sequencing screen (Table S1) of that tumour type
  miRNA drivers = OncomiR tumour-development miRNAs (p < 0.05) for that tumour
                  type; LGG has no normal tissue in OncomiR, so the pan-cancer
                  OncomiR list is used for LGG.

Screen selection (by Table S1 'Cancer type' text):
  BRCA : contains 'breast', excluding benign fibroadenoma / fibroepithelial tumours
  KIPAN: contains 'renal', excluding renal angiomyolipoma
  LGG  : adult glioma screens - 'low grade glioma', 'oligodendroglioma',
         'pan-glioma', 'glioma' (paediatric, DIPG, glioblastoma-only and
         pheochromocytoma/paraganglioma screens excluded)

Output: data/biological_priors/cancer_drivers_<cohort>.json
"""

# The per-cohort screen rules and OncomiR codes are read from configs/cohorts.json
# ("drivers" entry); the BRCA / KIPAN / LGG rules there are the ones written above.
# (This docstring is copied into every output file as "rule", so it is kept unchanged.)

import re
import json
import argparse
import pandas as pd

P = "data/biological_priors/raw_downloads/ncg7_paper_supplement/13059_2022_2607_"
OUT = "data/biological_priors"
COHORTS = {k: v for k, v in json.load(open("configs/cohorts.json")).items() if not k.startswith("_")}


def screen_filter(cancer_type: str, cohort: str) -> bool:
    """NCG Table S1 'Cancer type' text matches the cohort's rule: exact names
    (ncg_screen_exact) or contains any of ncg_screen_contains and none of ncg_screen_excludes."""
    t = str(cancer_type).lower()
    rule = COHORTS[cohort]["drivers"]
    if "ncg_screen_exact" in rule:
        return t.strip() in set(rule["ncg_screen_exact"])
    return (any(w in t for w in rule["ncg_screen_contains"])
            and not any(w in t for w in rule.get("ncg_screen_excludes", [])))


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="all", choices=list(COHORTS) + ["all"])
    ap.add_argument("--out", default=OUT, help="output folder (default data/biological_priors)")
    a = ap.parse_args()
    OUT = a.out
    s1 = pd.read_excel(P + "MOESM1_ESM.xlsx", sheet_name="Table S1", header=4)
    s1.columns = [str(c).replace("\n", " ").strip() for c in s1.columns]
    s1 = s1[s1["Screen type"].astype(str).str.contains("cancer", case=False)]
    s3 = pd.read_excel(P + "MOESM4_ESM.xlsx", sheet_name="Table S3", header=4)
    s3.columns = [str(c).replace("\n", " ").strip() for c in s3.columns]
    ref_col = "Reference (cancer screens)"
    gene_refs = {str(g).strip(): set(re.findall(r"\[(\d+)\]", str(r))) for g, r in zip(s3["Gene symbol"], s3[ref_col])}

    onc = pd.read_csv("data/biological_priors/raw_downloads/oncomir_tumor_vs_normal_all_cancers_p0.05.tsv", sep="\t")
    for cohort in (list(COHORTS) if a.cohort == "all" else [a.cohort]):
        sel = s1[s1["Cancer type"].apply(lambda t: screen_filter(t, cohort))]
        refs = set(re.findall(r"\[(\d+)\]", " ".join(sel["Reference"].astype(str))))
        genes = sorted(g for g, r in gene_refs.items() if r & refs)
        rule = COHORTS[cohort]["drivers"]
        codes = rule["oncomir_codes"]          # None = pan-cancer OncomiR list
        mirs = onc if codes is None else onc[onc["Cancer Abbreviation"].isin(codes)]
        mirnas = sorted(mirs["miRNA Name"].dropna().astype(str).unique())
        info = {"cohort": cohort.upper(), "genes": genes, "mirnas": mirnas,
                "screens_used": sel[["Publication", "Reference", "Cancer type"]].astype(str).values.tolist(),
                "oncomir_cancer_codes": codes or rule.get("oncomir_note", "all (pan-cancer OncomiR list)"),
                "rule": __doc__.strip()}
        with open(f"{OUT}/cancer_drivers_{cohort}.json", "w") as f:
            json.dump(info, f, indent=1)
        print(f"{cohort.upper()}: {len(sel)} screens -> {len(genes)} canonical driver genes; {len(mirnas)} OncomiR miRNAs")


if __name__ == "__main__":
    main()
