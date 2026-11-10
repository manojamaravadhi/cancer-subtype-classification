"""
Build the miRBase map used by src/preprocess_exact.py from the official download.

Output (in data/biological_priors/):
  mirbase_mimat_to_name.tsv   columns: mimat_id, mirna_name   (from miRBase hsa.gff3)

Nothing here invents entries: every row comes from the downloaded miRBase file.

Usage:
  python src/build_priors.py --mirbase data/biological_priors/mirbase_hsa_v23.1.gff3
"""

import os
import argparse
import pandas as pd

PRIORS_DIR = "data/biological_priors"


def build_mirbase_map(gff3_path: str) -> str:
    """Parse mature miRNA rows (type == 'miRNA') of a miRBase gff3 into MIMAT -> name."""
    rows = []
    with open(gff3_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9 or parts[2] != "miRNA":
                continue
            attrs = dict(kv.split("=", 1) for kv in parts[8].split(";") if "=" in kv)
            if "ID" in attrs and "Name" in attrs:
                rows.append((attrs["ID"], attrs["Name"]))
    df = pd.DataFrame(rows, columns=["mimat_id", "mirna_name"]).drop_duplicates("mimat_id")
    out = os.path.join(PRIORS_DIR, "mirbase_mimat_to_name.tsv")
    df.to_csv(out, sep="\t", index=False)
    print(f"miRBase map: {len(df)} MIMAT -> name rows written to {out} (source: {gff3_path})")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mirbase", required=True, help="Path to miRBase hsa.gff3")
    a = p.parse_args()
    os.makedirs(PRIORS_DIR, exist_ok=True)
    build_mirbase_map(a.mirbase)


if __name__ == "__main__":
    main()
