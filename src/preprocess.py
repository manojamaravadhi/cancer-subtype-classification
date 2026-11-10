"""
Helper functions used by src/preprocess_exact.py (barcode handling, sample alignment, matrix reading).

  norm_barcode  TCGA sample barcode -> 15 characters (e.g. TCGA-XX-XXXX-01)
  is_tumor      primary / metastatic tumour samples (sample type 01 or 06)
  read_matrix   read a Xena gzipped matrix (features x samples) restricted to given columns
  align_one     tumour samples present in mRNA, miRNA, methylation 450K and survival of one TCGA project;
                one sample per patient: a metastatic (06) sample is dropped when the same patient has a
                primary (01) sample (added 2026-11-09 with the owner's approval)
"""

import gzip
import pandas as pd


def norm_barcode(barcode: str) -> str:
    """Normalize TCGA sample barcode to 15 characters (e.g. TCGA-XX-XXXX-01)."""
    return str(barcode).strip()[:15]


def is_tumor(barcode: str) -> bool:
    """Primary / metastatic tumour samples (sample type 01 or 06)."""
    parts = str(barcode).strip().split("-")
    return len(parts) >= 4 and parts[3].startswith(("01", "06"))


def get_gz_header(filepath: str) -> list:
    with gzip.open(filepath, "rt") as f:
        return f.readline().strip().split("\t")


def read_matrix(path: str, cols: list, new_names: list) -> pd.DataFrame:
    """Read a Xena gzipped matrix (features x samples) restricted to `cols`."""
    df = pd.read_csv(path, sep="\t", usecols=["sample"] + cols).set_index("sample")[cols]
    df.columns = new_names
    df = df[~df.index.duplicated(keep="first")]
    return df


def align_one(pdir: str, p: str):
    """Return aligned common barcodes and raw column names for one TCGA project."""
    m_head = get_gz_header(f"{pdir}/{p}_mRNA_HiSeqV2.gz")
    mi_head = get_gz_header(f"{pdir}/{p}_miRNA_HiSeq.gz")
    me_head = get_gz_header(f"{pdir}/{p}_methylation450k.gz")
    surv = pd.read_csv(f"{pdir}/{p}_survival.tsv", sep="\t")
    m = {norm_barcode(s): s for s in m_head[1:] if is_tumor(s)}
    mi = {norm_barcode(s): s for s in mi_head[1:] if is_tumor(s)}
    me = {norm_barcode(s): s for s in me_head[1:] if is_tumor(s)}
    sv = {norm_barcode(str(s)): s for s in surv["sample"] if is_tumor(str(s))}
    common = sorted(set(m) & set(mi) & set(me) & set(sv))
    # one sample per patient: drop a metastatic (06) sample if the same patient has a primary (01) sample,
    # so that no patient can have samples in both a training and a test fold
    primary = {s[:12] for s in common if s[13:15] == "01"}
    dropped = [s for s in common if s[13:15] == "06" and s[:12] in primary]
    if dropped:
        print(f"  dropped {len(dropped)} metastatic samples of patients with a primary sample: {dropped}")
    common = [s for s in common if s not in dropped]
    surv["norm_sample"] = surv["sample"].apply(norm_barcode)
    surv = surv[surv["norm_sample"].isin(common)].drop_duplicates("norm_sample").set_index("norm_sample").loc[common]
    return common, [m[s] for s in common], [mi[s] for s in common], [me[s] for s in common], surv
