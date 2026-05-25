#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
07_plot_celltype_pwm_logos.py

Use CREsted official plotting functions from:
    crested.pl.modisco.selected_instances

This script reads:
    <species>/pattern_matrix/<species>_all_patterns.pkl
    <species>/celltype_pattern_matrix/<species>_top_patterns_per_celltype.tsv

Then for each cell type, it plots the top N pattern clusters using the official CREsted function.

Run:
    python 07_plot_celltype_pwm_logos.py --species both --top-n 5
"""

from pathlib import Path
import argparse
import pickle
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import crested


BASE = Path("/mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/pjt")
STAGE2 = BASE / "runs/out/stage2_topic_modisco_top1000"


def pattern_cluster_to_int(x):
    s = str(x).strip()
    s = s.replace("pattern_cluster_", "")
    return int(s)


def safe_name(x):
    return (
        str(x)
        .replace("/", "_")
        .replace(" ", "_")
        .replace(":", "_")
        .replace(",", "_")
        .replace("(", "")
        .replace(")", "")
    )


def run_species(species, top_n=5, agg="mean"):
    print(f"\n===== {species} =====")

    pattern_pkl = STAGE2 / species / "pattern_matrix" / f"{species}_all_patterns.pkl"
    cell_dir = STAGE2 / species / "celltype_pattern_matrix"
    outdir = STAGE2 / species / "celltype_pwm_logos"
    outdir.mkdir(parents=True, exist_ok=True)

    top_candidates = [
        cell_dir / f"{species}_top_patterns_per_celltype_{agg}.tsv",
        cell_dir / f"{species}_top_patterns_per_celltype.tsv",
    ]

    top_file = None
    for f in top_candidates:
        if f.exists():
            top_file = f
            break

    if top_file is None:
        raise FileNotFoundError(
            "Cannot find top pattern table. Tried:\n"
            + "\n".join(str(x) for x in top_candidates)
        )

    print("[patterns]", pattern_pkl)
    print("[top table]", top_file)

    with open(pattern_pkl, "rb") as f:
        all_patterns = pickle.load(f)

    df = pd.read_csv(top_file, sep="\t")

    required = ["celltype", "pattern_cluster", "rank"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column {col}. Existing columns: {list(df.columns)}")

    df = df[df["rank"] <= top_n].copy()
    df = df.sort_values(["celltype", "rank"])

    celltypes = list(df["celltype"].drop_duplicates())

    summary_rows = []

    for ct in celltypes:
        sub = df[df["celltype"] == ct].sort_values("rank").copy()

        pattern_indices = [pattern_cluster_to_int(x) for x in sub["pattern_cluster"]]
        print(f"[plot] {species} | {ct} | patterns = {pattern_indices}")

        try:
            crested.pl.modisco.selected_instances(all_patterns, pattern_indices)

            out_png = outdir / f"{species}_{safe_name(ct)}_top{top_n}_selected_instances.png"
            plt.savefig(out_png, dpi=300, bbox_inches="tight")
            plt.close("all")

            print("[saved]", out_png)

            for _, row in sub.iterrows():
                summary_rows.append({
                    "species": species,
                    "celltype": ct,
                    "rank": row["rank"],
                    "pattern_cluster": row["pattern_cluster"],
                    "pattern_index": pattern_cluster_to_int(row["pattern_cluster"]),
                    "output_png": str(out_png),
                })

        except Exception as e:
            plt.close("all")
            print(f"[ERROR] {species} {ct}: {type(e).__name__}: {e}")

    summary = pd.DataFrame(summary_rows)
    out_summary = outdir / f"{species}_selected_instances_summary_top{top_n}_{agg}.tsv"
    summary.to_csv(out_summary, sep="\t", index=False)
    print("[summary]", out_summary)

    # Also make one global figure with all unique top patterns.
    unique_patterns = sorted(set(summary["pattern_index"])) if not summary.empty else []
    if unique_patterns:
        print("[plot global unique patterns]", unique_patterns)
        try:
            crested.pl.modisco.selected_instances(all_patterns, unique_patterns)
            out_global = outdir / f"{species}_all_celltype_unique_top_patterns_top{top_n}_{agg}.png"
            plt.savefig(out_global, dpi=300, bbox_inches="tight")
            plt.close("all")
            print("[saved global]", out_global)
        except Exception as e:
            plt.close("all")
            print(f"[ERROR global] {type(e).__name__}: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--species", choices=["human", "macaque", "both"], default="both")
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--agg", choices=["mean", "max"], default="mean")
    args = parser.parse_args()

    if args.species == "both":
        for sp in ["human", "macaque"]:
            run_species(sp, top_n=args.top_n, agg=args.agg)
    else:
        run_species(args.species, top_n=args.top_n, agg=args.agg)


if __name__ == "__main__":
    main()
