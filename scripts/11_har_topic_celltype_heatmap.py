#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse
import re
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist


BASE = Path("/mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/pjt")
OUT = BASE / "runs/out"

SCORE_DIRS = {
    "humanModel": OUT / "har_3species_hscores",
    "macaqueModel": OUT / "har_3species_mscores",
}

ANNOT = {
    "humanModel": BASE / "data/stage0_annotation/topic_annotation_human.tsv",
    "macaqueModel": BASE / "data/stage0_annotation/topic_annotation_macaque.tsv",
}


def natural_topic_sort(cols):
    def key(x):
        m = re.match(r"Topic(\d+)$", str(x))
        return int(m.group(1)) if m else 10**9
    return sorted(cols, key=key)


def find_topic_cols(df):
    return natural_topic_sort([c for c in df.columns if re.match(r"Topic\d+$", str(c))])


def normalize_topic(x):
    x = str(x).strip()
    if x.startswith("Topic"):
        return x
    try:
        return f"Topic{int(float(x))}"
    except Exception:
        return x


def split_celltypes(x):
    if pd.isna(x):
        return []
    return [i.strip() for i in str(x).split(",") if i.strip()]


def load_topic_to_celltypes(annot_file):
    annot = pd.read_csv(annot_file, sep="\t")

    topic_col = "topic" if "topic" in annot.columns else annot.columns[0]

    if "final_class" in annot.columns:
        ct_col = "final_class"
    elif "celltype" in annot.columns:
        ct_col = "celltype"
    elif "top1_celltype" in annot.columns:
        ct_col = "top1_celltype"
    else:
        raise ValueError(f"Cannot find celltype column in {annot_file}. Columns: {annot.columns.tolist()}")

    topic_to_ct = {}

    for _, r in annot.iterrows():
        topic = normalize_topic(r[topic_col])

        if "is_general" in annot.columns:
            if str(r["is_general"]).lower() == "true":
                continue

        cts = split_celltypes(r[ct_col])
        if len(cts) > 0:
            topic_to_ct[topic] = cts

    return topic_to_ct


def build_celltype_matrix(df, topic_cols, topic_to_ct, agg="mean"):
    all_cts = sorted(set(ct for cts in topic_to_ct.values() for ct in cts))

    rows = {}

    for ct in all_cts:
        cols = [t for t in topic_cols if ct in topic_to_ct.get(t, [])]

        if len(cols) == 0:
            continue

        sub = df[cols]

        if agg == "max":
            rows[ct] = sub.max(axis=1)
        else:
            rows[ct] = sub.mean(axis=1)

    ct_df = pd.DataFrame(rows, index=df.index)
    return ct_df


def row_zscore(mat):
    arr = mat.to_numpy(dtype=float)
    mean = np.nanmean(arr, axis=1, keepdims=True)
    std = np.nanstd(arr, axis=1, keepdims=True)
    std[std == 0] = 1
    z = (arr - mean) / std
    return pd.DataFrame(z, index=mat.index, columns=mat.columns)


def cluster_order(mat, axis=0):
    arr = mat.to_numpy(dtype=float)

    if axis == 1:
        arr = arr.T

    if arr.shape[0] <= 2:
        return np.arange(arr.shape[0])

    arr = np.nan_to_num(arr)

    try:
        dist = pdist(arr, metric="correlation")
        link = linkage(dist, method="average")
        return leaves_list(link)
    except Exception:
        return np.arange(arr.shape[0])


def plot_heatmap(mat, out_png, title, row_cluster=True, col_cluster=True, figsize=(16, 12), vmax=None):
    plot_mat = mat.copy()

    if row_cluster:
        ro = cluster_order(plot_mat, axis=0)
        plot_mat = plot_mat.iloc[ro, :]

    if col_cluster:
        co = cluster_order(plot_mat, axis=1)
        plot_mat = plot_mat.iloc[:, co]

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(
        plot_mat.to_numpy(dtype=float),
        aspect="auto",
        interpolation="nearest",
        vmin=-vmax if vmax else None,
        vmax=vmax,
    )

    ax.set_title(title)
    ax.set_xlabel("Topics / cell types")
    ax.set_ylabel("HARs")

    ax.set_xticks(np.arange(plot_mat.shape[1]))
    ax.set_xticklabels(plot_mat.columns, rotation=90, fontsize=6)

    # HAR 太多，不显示全部 y label
    ax.set_yticks([])

    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("score")

    fig.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=250, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    print("[save]", out_png)
    print("[save]", out_png.with_suffix(".pdf"))


def process_one(model_label, species, agg="mean", transform="zscore"):
    score_dir = SCORE_DIRS[model_label]

    if species == "all":
        score_file = score_dir / f"all_3species_activeHAR_scores_{model_label}.csv"
    else:
        score_file = score_dir / f"{species}_activeHAR_scores_{model_label}.csv"

    if not score_file.exists():
        raise FileNotFoundError(score_file)

    outdir = OUT / "har_topic_celltype_heatmaps" / model_label / species
    outdir.mkdir(parents=True, exist_ok=True)

    print("\n==============================")
    print("[model]", model_label)
    print("[species]", species)
    print("[score]", score_file)

    df = pd.read_csv(score_file)

    topic_cols = find_topic_cols(df)
    print("[n HAR]", df.shape[0])
    print("[n topics]", len(topic_cols))

    # row id
    if species == "all" and "species" in df.columns:
        df.index = df["har_id"].astype(str) + "|" + df["species"].astype(str)
    else:
        df.index = df["har_id"].astype(str)

    topic_mat = df[topic_cols].copy()

    # 保存原始矩阵
    topic_mat.to_csv(outdir / f"{species}_{model_label}_har_by_topic_raw.tsv", sep="\t")

    if transform == "zscore":
        topic_plot = row_zscore(topic_mat)
        suffix = "rowZ"
        vmax = 2.5
    else:
        topic_plot = topic_mat
        suffix = "raw"
        vmax = None

    plot_heatmap(
        topic_plot,
        outdir / f"{species}_{model_label}_har_by_topic_heatmap_{suffix}.png",
        title=f"{species} HAR × topic scores ({model_label}, {suffix})",
        row_cluster=True,
        col_cluster=True,
        figsize=(18, 12),
        vmax=vmax,
    )

    # celltype matrix
    topic_to_ct = load_topic_to_celltypes(ANNOT[model_label])
    ct_mat = build_celltype_matrix(topic_mat, topic_cols, topic_to_ct, agg=agg)

    print("[n celltypes]", ct_mat.shape[1])

    ct_mat.to_csv(
        outdir / f"{species}_{model_label}_har_by_celltype_{agg}_raw.tsv",
        sep="\t",
    )

    if transform == "zscore":
        ct_plot = row_zscore(ct_mat)
        ct_suffix = f"{agg}_rowZ"
        ct_vmax = 2.5
    else:
        ct_plot = ct_mat
        ct_suffix = f"{agg}_raw"
        ct_vmax = None

    plot_heatmap(
        ct_plot,
        outdir / f"{species}_{model_label}_har_by_celltype_heatmap_{ct_suffix}.png",
        title=f"{species} HAR × cell type scores ({model_label}, {ct_suffix})",
        row_cluster=True,
        col_cluster=True,
        figsize=(14, 12),
        vmax=ct_vmax,
    )

    # top celltype summary
    top_ct = ct_mat.idxmax(axis=1)
    top_score = ct_mat.max(axis=1)

    summary = pd.DataFrame({
        "har_id": df["har_id"].values,
        "species": df["species"].values if "species" in df.columns else species,
        "top_celltype": top_ct.values,
        "top_celltype_score": top_score.values,
        "top_topic": df["top_topic"].values if "top_topic" in df.columns else "",
        "top_topic_score": df["top_score"].values if "top_score" in df.columns else "",
        "entropy": df["entropy"].values if "entropy" in df.columns else "",
    }, index=ct_mat.index)

    summary.to_csv(
        outdir / f"{species}_{model_label}_har_top_celltype_summary.tsv",
        sep="\t",
        index=False,
    )

    print("[save summary]", outdir / f"{species}_{model_label}_har_top_celltype_summary.tsv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["humanModel", "macaqueModel", "both"], default="humanModel")
    parser.add_argument("--species", choices=["human", "chimp", "macaque", "all", "all_species"], default="human")
    parser.add_argument("--agg", choices=["mean", "max"], default="mean")
    parser.add_argument("--transform", choices=["zscore", "raw"], default="zscore")
    args = parser.parse_args()

    models = ["humanModel", "macaqueModel"] if args.model == "both" else [args.model]

    if args.species == "all_species":
        species_list = ["human", "chimp", "macaque", "all"]
    else:
        species_list = [args.species]

    for model in models:
        for sp in species_list:
            process_one(model, sp, agg=args.agg, transform=args.transform)


if __name__ == "__main__":
    main()