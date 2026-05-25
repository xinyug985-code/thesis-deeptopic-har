#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd

HAR_ID = "HARsv2_1687"
CASE_DIR = Path("/mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/pjt/runs/out/har1687_case")

human_csv = CASE_DIR / f"{HAR_ID}__human__top_windows.csv"
macaque_csv = CASE_DIR / f"{HAR_ID}__macaque__top_windows.csv"

human_fa = CASE_DIR / f"{HAR_ID}__human__top_windows.fa"
macaque_fa = CASE_DIR / f"{HAR_ID}__macaque__top_windows.fa"
merged_fa = CASE_DIR / f"{HAR_ID}__all_top_windows.fa"


def write_fasta(df: pd.DataFrame, out_fa: Path):
    with open(out_fa, "w") as f:
        for _, r in df.iterrows():
            header = (
                f">{r['har_id']}|{r['species']}|{r['model_label']}|rank{r['window_rank']}"
                f"|{r['chrom']}:{r['genomic_start_0based']}-{r['genomic_end_0based_exclusive']}"
            )
            seq = str(r["sequence"]).upper()
            f.write(header + "\n")
            f.write(seq + "\n")


def main():
    h = pd.read_csv(human_csv)
    m = pd.read_csv(macaque_csv)

    write_fasta(h, human_fa)
    write_fasta(m, macaque_fa)
    write_fasta(pd.concat([h, m], axis=0, ignore_index=True), merged_fa)

    print("Saved:", human_fa)
    print("Saved:", macaque_fa)
    print("Saved:", merged_fa)


if __name__ == "__main__":
    main()