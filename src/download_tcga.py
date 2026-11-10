"""
Download the raw TCGA files of any cohort in configs/cohorts.json from the UCSC Xena
TCGA hub - five files per TCGA project:

  <P>_mRNA_HiSeqV2.gz        TCGA.<P>.sampleMap/HiSeqV2.gz
  <P>_miRNA_HiSeq.gz         TCGA.<P>.sampleMap/miRNA_HiSeq_gene.gz
  <P>_methylation450k.gz     TCGA.<P>.sampleMap/HumanMethylation450.gz
  <P>_clinicalMatrix.tsv     TCGA.<P>.sampleMap/<P>_clinicalMatrix
  <P>_survival.tsv           survival/<P>_survival.txt

Files that already exist are not downloaded again. For every file the size, SHA-256
checksum, URL and date are written to data/raw/download_manifest.json, so a later
re-download can be compared exactly.

Usage: python -m src.download_tcga --cohort <name|all>
"""

import os
import json
import time
import hashlib
import argparse
import urllib.request

XENA_HUB = "https://tcga.xenahubs.net/download"
MANIFEST = "data/raw/download_manifest.json"


def files_for(p):
    return [(f"{XENA_HUB}/TCGA.{p}.sampleMap%2FHiSeqV2.gz", f"{p}_mRNA_HiSeqV2.gz"),
            (f"{XENA_HUB}/TCGA.{p}.sampleMap%2FmiRNA_HiSeq_gene.gz", f"{p}_miRNA_HiSeq.gz"),
            (f"{XENA_HUB}/TCGA.{p}.sampleMap%2FHumanMethylation450.gz", f"{p}_methylation450k.gz"),
            (f"{XENA_HUB}/TCGA.{p}.sampleMap%2F{p}_clinicalMatrix", f"{p}_clinicalMatrix.tsv"),
            (f"{XENA_HUB}/survival%2F{p}_survival.txt", f"{p}_survival.tsv")]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    cohorts = {k: v for k, v in json.load(open("configs/cohorts.json")).items() if not k.startswith("_")}
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="all", choices=list(cohorts) + ["all"])
    a = ap.parse_args()
    manifest = json.load(open(MANIFEST)) if os.path.exists(MANIFEST) else {}
    for c in (list(cohorts) if a.cohort == "all" else [a.cohort]):
        for p, pdir in cohorts[c]["projects"].items():
            os.makedirs(pdir, exist_ok=True)
            for url, name in files_for(p):
                dest = os.path.join(pdir, name)
                key = dest.replace("\\", "/")
                if os.path.exists(dest) and os.path.getsize(dest) > 0:
                    status = "already present"
                else:
                    print(f"  downloading {name} ...", flush=True)
                    tmp = dest + ".part"
                    urllib.request.urlretrieve(url, tmp)
                    os.replace(tmp, dest)
                    status = "downloaded"
                if key not in manifest or manifest[key].get("size") != os.path.getsize(dest):
                    manifest[key] = {"url": url, "size": os.path.getsize(dest), "sha256": sha256(dest),
                                     "recorded": time.strftime("%Y-%m-%d %H:%M:%S"), "status": status}
                print(f"  [{c}] {name}: {status} ({os.path.getsize(dest) / 1e6:.1f} MB)")
    with open(MANIFEST, "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"manifest: {MANIFEST}")


if __name__ == "__main__":
    main()
