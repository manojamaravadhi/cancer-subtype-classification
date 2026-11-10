"""
Preprocessing that follows the two reference papers (label-free steps only).

Cohorts, raw folders and label sources come from configs/cohorts.json (any cohort).
Common: same tumour samples as before (mRNA + miRNA + 450K + survival), one sample per patient
        (a metastatic 06 sample is dropped when the patient has a primary 01 sample; src/preprocess.py).
Labels : BRCA - PAM50Call_RNAseq (Xena); KIPAN - KIRC/KIRP/KICH project;
         LGG  - TCGA PanCancer Atlas SUBTYPE from cBioPortal
                (lgg_tcga_pan_can_atlas_2018, the source reference graph method used):
                LGG_IDHmut-codel / LGG_IDHmut-non-codel / LGG_IDHwt.

Gene-level methylation (reference transformer paper Sec. 4.2): CpGs annotated TSS1500 in the Illumina
450K manifest (GEO GPL13534), after removing non-CpG, X/Y and SNP probes
(ChAMP-style), probes missing in >25% of samples removed, then the mean of a
gene's TSS1500 CpGs = that gene's methylation value. Remaining gaps are filled
with KNN imputation over genes (impute.knn style, k = 10), which uses only the
same sample's other genes - no information is shared between patients.

Stream A inputs (reference transformer method 4.2): mRNA and miRNA features that are 0 in > 25% of
  samples removed; methylation as above.
Stream B inputs (reference graph method data_process.py): features with > 10% missing removed,
  remaining missing values set to 0, then features with >= 10% zeros removed.

Per-fold steps (scaling, L1-SVC, z-scores) happen later in src/train_exact.py.
Output: data/processed_exact/tcga_<cohort>/
"""

import os
import json
import time
import argparse
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

from src.preprocess import align_one, norm_barcode, read_matrix

OUT = "data/processed_exact"
COHORTS = "configs/cohorts.json"     # which cohorts exist, their raw folders and label sources
TSS = "data/biological_priors/hm450_tss1500_probe_to_gene.tsv"
MIRBASE = "data/biological_priors/mirbase_mimat_to_name.tsv"


def gene_level_methylation(paths_cols, samples):
    """Read only TSS1500 probes, drop probes missing in >25%, average per gene."""
    tss = pd.read_csv(TSS, sep="\t")
    probes = set(tss["probe"])
    parts = []
    for path, cols, names in paths_cols:
        kept = []
        for chunk in pd.read_csv(path, sep="\t", usecols=["sample"] + cols, chunksize=40000):
            chunk = chunk.set_index("sample")[cols]
            chunk = chunk[chunk.index.isin(probes) & ~chunk.index.duplicated()].astype(np.float32)
            chunk.columns = names
            kept.append(chunk)
        parts.append(pd.concat(kept))
    common = sorted(set.intersection(*[set(p.index) for p in parts]))
    cpg = pd.concat([p.loc[common] for p in parts], axis=1)[samples]
    n_cpg = len(cpg)
    cpg = cpg[cpg.isna().mean(axis=1) <= 0.25]
    t = tss[tss["probe"].isin(cpg.index)]
    gene = cpg.loc[t["probe"]].set_index(t["gene"].values).groupby(level=0).mean()   # mean over a gene's CpGs
    n_nan = int(gene.isna().sum().sum())
    if n_nan:
        imp = KNNImputer(n_neighbors=10)                                   # rows = genes (impute.knn style)
        gene = pd.DataFrame(imp.fit_transform(gene.values), index=gene.index, columns=gene.columns)
    print(f"    methylation: {n_cpg} TSS1500 CpGs -> {len(cpg)} after 25% missing filter -> "
          f"{len(gene)} genes ({n_nan} missing gene values imputed)")
    return gene


def trans_filter(df):
    """reference transformer method: remove features that are 0 in > 25% of samples (missing counted as 0)."""
    d = df.fillna(0.0)
    return d[(d == 0).mean(axis=1) <= 0.25]


def graph_filter(df):
    """reference graph method data_process.py: drop >10% NA, fill 0, keep features with < 10% zeros."""
    d = df[df.isna().mean(axis=1) <= 0.10].fillna(0.0)
    return d[(d == 0).mean(axis=1) < 0.10]


def save(out_dir, name, samples, raw, meth_gene, y, ev, tm, classes, extra):
    os.makedirs(out_dir, exist_ok=True)
    mb = pd.read_csv(MIRBASE, sep="\t")
    mimat2name = dict(zip(mb["mimat_id"], mb["mirna_name"]))
    mats = {
        "trans_mrna": trans_filter(raw["mrna"]), "trans_mirna": trans_filter(raw["mirna"]),
        "trans_meth": meth_gene,
        "graph_mrna": graph_filter(raw["mrna"]), "graph_mirna": graph_filter(raw["mirna"]),
        "graph_meth": graph_filter(meth_gene),
    }
    names = {}
    for k, df in mats.items():
        np.save(os.path.join(out_dir, f"X_{k}.npy"), df.T.values.astype(np.float32))
        idx = list(df.index)
        names[k] = [mimat2name.get(i, i) for i in idx] if k.endswith("mirna") else idx
    with open(os.path.join(out_dir, "feature_names.json"), "w") as f:
        json.dump(names, f)
    np.save(os.path.join(out_dir, "y_subtype.npy"), np.asarray(y, dtype=np.int64))
    np.save(os.path.join(out_dir, "y_os_event.npy"), np.asarray(ev, dtype=np.int64))
    np.save(os.path.join(out_dir, "y_os_time.npy"), np.asarray(tm, dtype=np.float32))
    np.save(os.path.join(out_dir, "sample_ids.npy"), np.array(samples))
    y = np.asarray(y)
    meta = {"cohort": name, "num_samples": len(samples), "class_names": classes,
            "subtype_distribution": {c: int((y == i).sum()) for i, c in enumerate(classes)},
            "subtype_unlabeled": int((y < 0).sum()), "survival_events": int(np.sum(ev)),
            "features": {k: int(len(v)) for k, v in names.items()}, **extra}
    with open(os.path.join(out_dir, "preprocessing_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps({k: meta[k] for k in ["num_samples", "subtype_distribution", "subtype_unlabeled",
                                           "survival_events", "features"]}, indent=1))


def load_cohorts(path=COHORTS):
    """Cohort registry (configs/cohorts.json); keys starting with '_' are notes."""
    return {k: v for k, v in json.load(open(path)).items() if not k.startswith("_")}


def labels_for(spec, common, pdir, p):
    """Subtype labels of one TCGA project's aligned samples, from the cohort's label source."""
    lab_spec = spec["labels"]
    classes = lab_spec["classes"]
    if lab_spec["type"] == "xena_clinical":
        clin = pd.read_csv(f"{pdir}/{p}_clinicalMatrix.tsv", sep="\t")
        clin["n"] = clin["sampleID"].apply(norm_barcode)
        v = clin.drop_duplicates("n").set_index("n").reindex(common)[lab_spec["column"]].values
    elif lab_spec["type"] == "file":
        lab = pd.read_csv(lab_spec["path"], sep="\t").drop_duplicates(lab_spec["id_column"])
        m = dict(zip(lab[lab_spec["id_column"]], lab[lab_spec["label_column"]]))
        v = np.array([m.get(s[:lab_spec["id_length"]]) for s in common], dtype=object)
    else:
        raise ValueError(f"label type {lab_spec['type']} needs one TCGA project per class")
    mp = {c: i for i, c in enumerate(classes)}
    return classes, np.array([mp.get(x, -1) for x in v]), lab_spec["source"]


def process_single(cohort, spec):
    """Cohort made of one TCGA project (e.g. BRCA, LGG)."""
    (p, pdir), = spec["projects"].items()
    print("\n" + "=" * 70 + f"\n{spec['name']}\n" + "=" * 70)
    t0 = time.time()
    common, mc, mic, mec, surv = align_one(pdir, p)
    classes, y, src = labels_for(spec, common, pdir, p)
    print(f"  samples {len(common)}, labeled {(y >= 0).sum()}")
    raw = {"mrna": read_matrix(f"{pdir}/{p}_mRNA_HiSeqV2.gz", mc, common),
           "mirna": read_matrix(f"{pdir}/{p}_miRNA_HiSeq.gz", mic, common)}
    meth = gene_level_methylation([(f"{pdir}/{p}_methylation450k.gz", mec, common)], common)
    save(f"{OUT}/tcga_{cohort}", spec["name"], common, raw, meth, y, surv["OS"].fillna(0).astype(int).values,
         surv["OS.time"].fillna(0.0).astype(float).values, classes,
         {"label_source": src, "elapsed_seconds": round(time.time() - t0, 1)})


def process_merged(cohort, spec):
    """Cohort merged from several TCGA projects (e.g. KIPAN = KIRC + KIRP + KICH):
    features = those present in every project; samples in project order."""
    projects = list(spec["projects"])
    print("\n" + "=" * 70 + f"\n{spec['name']}\n" + "=" * 70)
    t0 = time.time()
    subs = {}
    for p, pdir in spec["projects"].items():
        common, mc, mic, mec, surv = align_one(pdir, p)
        subs[p] = dict(dir=pdir, common=common, mc=mc, mic=mic, mec=mec, surv=surv)
    samples = [s for p in projects for s in subs[p]["common"]]
    if spec["labels"]["type"] == "project":
        classes = spec["labels"]["classes"]
        y = np.concatenate([[classes.index(p)] * len(subs[p]["common"]) for p in projects])
        src = spec["labels"]["source"]
    else:
        parts = [labels_for(spec, subs[p]["common"], subs[p]["dir"], p) for p in projects]
        classes, src = parts[0][0], parts[0][2]
        y = np.concatenate([part[1] for part in parts])
    ev = np.concatenate([subs[p]["surv"]["OS"].fillna(0).astype(int).values for p in projects])
    tm = np.concatenate([subs[p]["surv"]["OS.time"].fillna(0.0).astype(float).values for p in projects])

    def merged(kind, key):
        dfs = [read_matrix(f"{subs[p]['dir']}/{p}_{kind}.gz", subs[p][key], subs[p]["common"]) for p in projects]
        feats = sorted(set.intersection(*[set(d.index) for d in dfs]))
        return pd.concat([d.loc[feats] for d in dfs], axis=1)[samples]
    raw = {"mrna": merged("mRNA_HiSeqV2", "mc"), "mirna": merged("miRNA_HiSeq", "mic")}
    meth = gene_level_methylation([(f"{subs[p]['dir']}/{p}_methylation450k.gz", subs[p]["mec"], subs[p]["common"])
                                   for p in projects], samples)
    save(f"{OUT}/tcga_{cohort}", spec["name"], samples, raw, meth, y, ev, tm, classes,
         {"label_source": src, "elapsed_seconds": round(time.time() - t0, 1)})


def main():
    global OUT
    cohorts = load_cohorts()
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="all", choices=list(cohorts) + ["all"])
    ap.add_argument("--out", default=OUT, help="output root (default data/processed_exact)")
    a = ap.parse_args()
    OUT = a.out
    for c in (list(cohorts) if a.cohort == "all" else [a.cohort]):
        spec = cohorts[c]
        process_merged(c, spec) if len(spec["projects"]) > 1 else process_single(c, spec)


if __name__ == "__main__":
    main()
