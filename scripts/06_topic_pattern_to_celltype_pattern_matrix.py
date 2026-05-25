#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


BASE = Path("/mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/pjt")
STAGE2 = BASE / "runs" / "out" / "stage2_topic_modisco_top1000"

DATASETS = {
    "human": {
        "input": STAGE2 / "human" / "pattern_matrix" / "human_topic_by_pattern_matrix_with_annotation.tsv",
    },
    "macaque": {
        "input": STAGE2 / "macaque" / "pattern_matrix" / "macaque_topic_by_pattern_matrix_with_annotation.tsv",
    },
}


def split_celltypes(x):
    if pd.isna(x):
        return []
    return [i.strip() for i in str(x).split(",") if i.strip()]


def find_pattern_cols(df):
    return [c for c in df.columns if c.startswith("pattern_cluster_")]


def topic_to_celltype_matrix(df, agg="mean"):
    pattern_cols = find_pattern_cols(df)

    if "final_class" not in df.columns:
        raise ValueError("Missing final_class column")

    rows = []

    for _, row in df.iterrows():
        topic = row["topic"]
        cts = split_celltypes(row["final_class"])

        for ct in cts:
            r = {
                "celltype": ct,
                "topic": topic,
            }
            for c in pattern_cols:
                r[c] = row[c]
            rows.append(r)

    long_df = pd.DataFrame(rows)

    if long_df.empty:
        raise ValueError("No celltype-topic mapping found")

    if agg == "max":
        cell_df = long_df.groupby("celltype")[pattern_cols].max()
    else:
        cell_df = long_df.groupby("celltype")[pattern_cols].mean()

    return cell_df, long_df


def filter_patterns(cell_df, min_max_value=1.0):
    keep = cell_df.columns[cell_df.max(axis=0) >= min_max_value]
    return cell_df[keep]


def zscore_columns(df):
    x = df.copy()
    return (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-8)


def plot_heatmap(cell_df, out_png, out_pdf, title):
    cell_df = filter_patterns(cell_df, min_max_value=1.0)

    # column-wise zscore
    z = zscore_columns(cell_df)
    z = z.replace([np.inf, -np.inf], np.nan).fillna(0)

    height = max(6, 0.45 * z.shape[0])
    width = max(12, 0.28 * z.shape[1])

    cg = sns.clustermap(
        z,
        cmap="vlag",
        center=0,
        figsize=(width, height),
        linewidths=0.2,
        linecolor="lightgray",
        row_cluster=True,
        col_cluster=True,
        xticklabels=True,
        yticklabels=True,
        cbar_kws={"label": "Pattern importance z-score"},
    )

    cg.fig.suptitle(title, y=1.02)
    cg.ax_heatmap.set_xlabel("Pattern clusters")
    cg.ax_heatmap.set_ylabel("Cell types")

    cg.fig.savefig(out_png, dpi=250, bbox_inches="tight")
    cg.fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(cg.fig)


def summarize_top_patterns(cell_df, species, outdir, top_n=10):
    rows = []

    for ct in cell_df.index:
        s = cell_df.loc[ct].sort_values(ascending=False).head(top_n)

        for rank, (pattern, value) in enumerate(s.items(), start=1):
            rows.append({
                "species": species,
                "celltype": ct,
                "rank": rank,
                "pattern_cluster": pattern,
                "importance": value,
            })

    out = outdir / f"{species}_top_patterns_per_celltype.tsv"
    pd.DataFrame(rows).to_csv(out, sep="\t", index=False)
    print("[save]", out)


def run_species(species, agg="mean"):
    cfg = DATASETS[species]
    infile = cfg["input"]

    outdir = STAGE2 / species / "celltype_pattern_matrix"
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"\n===== {species} =====")
    print("[input]", infile)

    df = pd.read_csv(infile, sep="\t")
    print("[topic matrix]", df.shape)

    cell_df, long_df = topic_to_celltype_matrix(df, agg=agg)

    print("[celltype matrix]", cell_df.shape)
    print("[celltypes]", list(cell_df.index))

    out_matrix = outdir / f"{species}_celltype_by_pattern_matrix_{agg}.tsv"
    out_long = outdir / f"{species}_topic_celltype_pattern_long.tsv"

    cell_df.to_csv(out_matrix, sep="\t")
    long_df.to_csv(out_long, sep="\t", index=False)

    print("[save]", out_matrix)
    print("[save]", out_long)

    summarize_top_patterns(cell_df, species, outdir)

    plot_heatmap(
        cell_df,
        out_png=outdir / f"{species}_celltype_pattern_heatmap_{agg}.png",
        out_pdf=outdir / f"{species}_celltype_pattern_heatmap_{agg}.pdf",
        title=f"{species} celltype × pattern matrix ({agg})",
    )

    print("[done]", species)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--species", choices=["human", "macaque", "both"], default="both")
    parser.add_argument("--agg", choices=["mean", "max"], default="mean")
    args = parser.parse_args()

    if args.species == "both":
        for sp in ["human", "macaque"]:
            run_species(sp, agg=args.agg)
    else:
        run_species(args.species, agg=args.agg)


if __name__ == "__main__":
    main()