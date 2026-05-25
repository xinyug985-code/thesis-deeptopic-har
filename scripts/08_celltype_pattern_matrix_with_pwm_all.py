#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
08_celltype_pattern_matrix_with_pwm_all.py

Use CREsted official function:
    crested.pl.modisco.clustermap_with_pwm_logos()

Run all:
    python scripts/08_celltype_pattern_matrix_with_pwm_all.py

Run only human:
    python scripts/08_celltype_pattern_matrix_with_pwm_all.py --species human

Run only max:
    python scripts/08_celltype_pattern_matrix_with_pwm_all.py --agg max
"""
import os
os.environ["MPLBACKEND"] = "Agg"

from pathlib import Path
import argparse
import pickle
import pandas as pd

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

import crested


BASE = Path("/mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/pjt")
STAGE2 = BASE / "runs/out/stage2_topic_modisco_top1000"


def pat_idx(x):
    return int(str(x).replace("pattern_cluster_", ""))


def run_one(species, agg):
    print(f"\n===== {species} | {agg} =====")

    pattern_pkl = STAGE2 / species / "pattern_matrix" / f"{species}_all_patterns.pkl"
    matrix_file = STAGE2 / species / "celltype_pattern_matrix" / f"{species}_celltype_by_pattern_matrix_{agg}.tsv"

    outdir = STAGE2 / species / "celltype_pattern_matrix_with_pwm"
    outdir.mkdir(parents=True, exist_ok=True)

    print("[patterns]", pattern_pkl)
    print("[matrix]", matrix_file)

    if not pattern_pkl.exists():
        raise FileNotFoundError(pattern_pkl)

    if not matrix_file.exists():
        raise FileNotFoundError(matrix_file)

    with open(pattern_pkl, "rb") as f:
        all_patterns = pickle.load(f)

    df = pd.read_csv(matrix_file, sep="\t", index_col=0)

    # columns should be pattern_cluster_0, pattern_cluster_1...
    df = df[sorted(df.columns, key=pat_idx)]

    celltypes = list(df.index)
    pattern_matrix = df.values.astype(float)

    print("[matrix shape]", pattern_matrix.shape)
    print("[n celltypes]", len(celltypes))
    print("[n patterns]", len(df.columns))

    crested.pl.modisco.clustermap_with_pwm_logos(
        pattern_matrix,
        classes=celltypes,
        pattern_dict=all_patterns,
        width=60,
        height=8,
        grid=True,
        dendrogram_ratio=(0.03, 0.15),
        logo_x_multiplier=1,
        logo_height_fraction=0.35,
        logo_y_padding=0.25,
    )

    out_png = outdir / f"{species}_celltype_by_pattern_matrix_{agg}_with_pwm.png"

    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close("all")

    print("[saved]", out_png)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--species", choices=["human", "macaque", "both"], default="both")
    parser.add_argument("--agg", choices=["mean", "max", "both"], default="both")
    args = parser.parse_args()

    species_list = ["human", "macaque"] if args.species == "both" else [args.species]
    agg_list = ["mean", "max"] if args.agg == "both" else [args.agg]

    for species in species_list:
        for agg in agg_list:
            try:
                run_one(species, agg)
            except Exception as e:
                print(f"[ERROR] {species} | {agg}: {type(e).__name__}: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
