#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse
import pickle
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.textpath import TextPath
from matplotlib.patches import PathPatch
from matplotlib.transforms import Affine2D
from matplotlib.font_manager import FontProperties

from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram
from scipy.spatial.distance import pdist


BASE = Path("/mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/pjt")
STAGE2 = BASE / "runs" / "out" / "stage2_topic_modisco_top1000"

DATASETS = {
    "human": {
        "input": STAGE2 / "human" / "pattern_matrix" / "human_topic_by_pattern_matrix_with_annotation.tsv",
        "patterns": STAGE2 / "human" / "pattern_matrix" / "human_all_patterns.pkl",
    },
    "macaque": {
        "input": STAGE2 / "macaque" / "pattern_matrix" / "macaque_topic_by_pattern_matrix_with_annotation.tsv",
        "patterns": STAGE2 / "macaque" / "pattern_matrix" / "macaque_all_patterns.pkl",
    },
}

BASES = ["A", "C", "G", "T"]
BASE_COLORS = {
    "A": "#2ca02c",
    "C": "#1f77b4",
    "G": "#ff7f0e",
    "T": "#d62728",
}


def split_celltypes(x):
    if pd.isna(x):
        return []
    return [i.strip() for i in str(x).split(",") if i.strip()]


def find_pattern_cols(df):
    return [c for c in df.columns if c.startswith("pattern_cluster_")]


def pat_idx(x):
    return int(str(x).replace("pattern_cluster_", ""))


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

    cell_df = cell_df[sorted(cell_df.columns, key=pat_idx)]

    return cell_df, long_df


def filter_patterns(cell_df, min_max_value=1.0):
    if min_max_value <= 0:
        return cell_df

    keep = cell_df.columns[cell_df.abs().max(axis=0) >= min_max_value]
    return cell_df[keep]


def zscore_columns(df):
    x = df.copy()
    return (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-8)


def cluster_order(mat, axis=0):
    if axis == 0:
        x = mat
    else:
        x = mat.T

    if x.shape[0] <= 1:
        return np.arange(x.shape[0])

    dist = pdist(x, metric="correlation")
    dist = np.nan_to_num(dist, nan=0.0, posinf=0.0, neginf=0.0)

    Z = linkage(dist, method="average")
    return leaves_list(Z)


def normalize_pwm(x):
    pwm = np.asarray(x, dtype=float)

    if pwm.ndim != 2:
        raise ValueError(f"PWM must be 2D, got {pwm.shape}")

    if pwm.shape[0] == 4 and pwm.shape[1] != 4:
        pwm = pwm.T

    if pwm.shape[1] != 4:
        raise ValueError(f"Expected L x 4 PWM, got {pwm.shape}")

    pwm = np.nan_to_num(pwm, nan=0.0, posinf=0.0, neginf=0.0)
    pwm = np.clip(pwm, 0, None)

    row_sum = pwm.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0

    return pwm / row_sum


def get_pattern_pwm(all_patterns, pattern_cluster):
    key = str(pat_idx(pattern_cluster))

    if key not in all_patterns:
        raise KeyError(f"Missing pattern key {key}")

    outer = all_patterns[key]

    if "instances" not in outer:
        raise KeyError(f"No instances in pattern {key}")

    instances = outer["instances"]

    best_key = max(
        instances.keys(),
        key=lambda k: instances[k].get("n_seqlets", 0),
    )

    inst = instances[best_key]

    if "ppm" in inst:
        pwm = normalize_pwm(inst["ppm"])
    elif "sequence" in inst:
        pwm = normalize_pwm(inst["sequence"])
    else:
        raise KeyError(f"No ppm/sequence in instance {best_key}")

    return pwm, best_key


def information_content(pwm):
    eps = 1e-8
    entropy = -np.sum(pwm * np.log2(pwm + eps), axis=1)
    return np.clip(2.0 - entropy, 0, 2)


def trim_pwm_by_ic(pwm, threshold=0.15):
    ic = information_content(pwm)
    keep = np.where(ic >= threshold)[0]

    if len(keep) == 0:
        return pwm

    return pwm[keep[0]:keep[-1] + 1]


def draw_logo(ax, pwm):
    pwm = trim_pwm_by_ic(pwm, threshold=0.15)
    ic = information_content(pwm)
    heights = pwm * ic[:, None]

    fp = FontProperties(family="DejaVu Sans", weight="bold")

    for x in range(pwm.shape[0]):
        order = np.argsort(heights[x])
        y = 0.0

        for j in order:
            h = float(heights[x, j])
            if h <= 0.01:
                continue

            base = BASES[j]
            tp = TextPath((0, 0), base, size=1, prop=fp)
            bb = tp.get_extents()

            sx = 0.85 / max(bb.width, 1e-6)
            sy = h / max(bb.height, 1e-6)

            trans = Affine2D().scale(sx, sy).translate(x + 0.05, y)

            patch = PathPatch(
                tp,
                transform=trans + ax.transData,
                color=BASE_COLORS[base],
                lw=0,
            )
            ax.add_patch(patch)

            y += h

    ax.set_xlim(0, pwm.shape[0])
    ax.set_ylim(0, 2.0)
    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(False)

def draw_logo_vertical(ax, pwm):
    pwm = trim_pwm_by_ic(pwm, threshold=0.15)
    ic = information_content(pwm)
    heights = pwm * ic[:, None]

    fp = FontProperties(family="DejaVu Sans", weight="bold")

    for y_pos in range(pwm.shape[0]):
        order = np.argsort(heights[y_pos])
        x_stack = 0.0

        for j in order:
            h = float(heights[y_pos, j])
            if h <= 0.01:
                continue

            base = BASES[j]
            tp = TextPath((0, 0), base, size=1, prop=fp)
            bb = tp.get_extents()

            sx = h / max(bb.width, 1e-6)
            sy = 0.85 / max(bb.height, 1e-6)

            trans = (
                Affine2D()
                .scale(sx, sy)
                .rotate_deg(90)
                .translate(x_stack + h, y_pos + 0.05)
            )

            patch = PathPatch(
                tp,
                transform=trans + ax.transData,
                color=BASE_COLORS[base],
                lw=0,
            )
            ax.add_patch(patch)

            x_stack += h

    ax.set_xlim(0, 2.0)
    ax.set_ylim(0, pwm.shape[0])
    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(False)

def get_linkage_and_order(mat, axis=0):
    x = mat if axis == 0 else mat.T

    if x.shape[0] <= 1:
        return None, np.arange(x.shape[0])

    dist = pdist(x, metric="correlation")
    dist = np.nan_to_num(dist, nan=0.0, posinf=0.0, neginf=0.0)

    Z = linkage(dist, method="average")
    order = leaves_list(Z)

    return Z, order

def plot_heatmap_with_pwm(cell_df, all_patterns, out_png, title, min_max_value=1.0, max_patterns=20):
    filtered = filter_patterns(cell_df, min_max_value=min_max_value)

    if filtered.shape[1] == 0:
        raise ValueError("No pattern left after filtering. Lower --min-max-value")

    score = filtered.abs().max(axis=0).sort_values(ascending=False)
    keep_cols = score.head(max_patterns).index
    filtered = filtered[keep_cols]

    z = zscore_columns(filtered)
    z = z.replace([np.inf, -np.inf], np.nan).fillna(0)

    # pattern × celltype
    z = z.T

    mat = z.values.astype(float)

    row_Z, row_order = get_linkage_and_order(mat, axis=0)
    col_Z, col_order = get_linkage_and_order(mat, axis=1)

    z_ord = z.iloc[row_order, col_order]
    pattern_rows = list(z_ord.index)
    celltypes = list(z_ord.columns)

    n_rows, n_cols = z_ord.shape

    fig_w = max(12, 0.75 * n_cols + 4)
    fig_h = max(8, 0.55 * n_rows)

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.suptitle(title, y=0.92, fontsize=14)

    gs = GridSpec(
        2,
        3,
        width_ratios=[3.0, 0.5, max(6, 0.75 * n_cols)],
        height_ratios=[0.7, max(5, 0.55 * n_rows)],
        hspace=0.03,
        wspace=0.035,
    )

    ax_col_dend = fig.add_subplot(gs[0, 2])
    ax_row_dend = fig.add_subplot(gs[1, 1])
    ax_logo_panel = gs[1, 0].subgridspec(n_rows, 1, hspace=0.05)
    ax_hm = fig.add_subplot(gs[1, 2])

    if col_Z is not None:
        dendrogram(
            col_Z,
            ax=ax_col_dend,
            orientation="top",
            no_labels=True,
            color_threshold=0,
            above_threshold_color="0.45",
            link_color_func=lambda k: "0.45",
        )
    ax_col_dend.axis("off")

    if row_Z is not None:
        dendrogram(
            row_Z,
            ax=ax_row_dend,
            orientation="left",
            no_labels=True,
            color_threshold=0,
            above_threshold_color="0.45",
            link_color_func=lambda k: "0.45",
        )
    ax_row_dend.axis("off")

    for ax in [ax_col_dend, ax_row_dend]:
        for line in ax.collections:
            line.set_linewidth(0.6)
            line.set_color("0.45")

    vmax = np.nanpercentile(np.abs(z_ord.values), 98)
    if vmax <= 0:
        vmax = 1

    im = ax_hm.imshow(
        z_ord.values,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        interpolation="nearest",
    )

    ax_hm.set_xlabel("Cell types")
    ax_hm.set_ylabel("Pattern clusters")

    ax_hm.set_xticks(np.arange(n_cols))
    ax_hm.set_xticklabels(celltypes, rotation=90, fontsize=8)

    ax_hm.set_yticks(np.arange(n_rows))
    ax_hm.set_yticklabels(pattern_rows, fontsize=8)

    ax_hm.yaxis.tick_right()
    ax_hm.yaxis.set_label_position("right")

    for x in np.arange(-0.5, n_cols, 1):
        ax_hm.axvline(x, color="lightgray", lw=0.25)
    for y in np.arange(-0.5, n_rows, 1):
        ax_hm.axhline(y, color="lightgray", lw=0.25)

    cax = fig.add_axes([0.08, 0.008, 0.012, 0.18])  # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Pattern importance z-score", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label("Pattern importance z-score")

    selected_rows = []

    for i, pat in enumerate(pattern_rows):
        ax_logo = fig.add_subplot(ax_logo_panel[i, 0])

        try:
            pwm, inst_key = get_pattern_pwm(all_patterns, pat)
            draw_logo(ax_logo, pwm)

            selected_rows.append({
                "pattern_cluster": pat,
                "pattern_index": pat_idx(pat),
                "selected_instance": inst_key,
                "pwm_length": pwm.shape[0],
                "max_abs_importance": float(filtered[pat].abs().max()),
            })

        except Exception as e:
            ax_logo.axis("off")
            ax_logo.text(0.5, 0.5, "ERR", ha="center", va="center", fontsize=6)
            selected_rows.append({
                "pattern_cluster": pat,
                "pattern_index": pat_idx(pat),
                "selected_instance": f"ERROR: {type(e).__name__}: {e}",
                "pwm_length": np.nan,
                "max_abs_importance": float(filtered[pat].abs().max()),
            })

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return selected_rows


def run_species(species, agg="mean", min_max_value=1.0):
    cfg = DATASETS[species]
    infile = cfg["input"]
    pattern_pkl = cfg["patterns"]

    outdir = STAGE2 / species / "celltype_pattern_matrix2"
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"\n===== {species} | {agg} =====")
    print("[input]", infile)
    print("[patterns]", pattern_pkl)

    df = pd.read_csv(infile, sep="\t")
    print("[topic matrix]", df.shape)

    with open(pattern_pkl, "rb") as f:
        all_patterns = pickle.load(f)

    cell_df, long_df = topic_to_celltype_matrix(df, agg=agg)

    print("[celltype matrix]", cell_df.shape)
    print("[celltypes]", list(cell_df.index))

    out_matrix = outdir / f"{species}_celltype_by_pattern_matrix_{agg}.tsv"
    out_long = outdir / f"{species}_topic_celltype_pattern_long_{agg}.tsv"

    cell_df.to_csv(out_matrix, sep="\t")
    long_df.to_csv(out_long, sep="\t", index=False)

    print("[save]", out_matrix)
    print("[save]", out_long)

    out_png = outdir / f"{species}_celltype_pattern_heatmap_{agg}_with_pwm_min{min_max_value}.png"

    selected_rows = plot_heatmap_with_pwm(
        cell_df,
        all_patterns=all_patterns,
        out_png=out_png,
        title=f"{species} celltype × pattern matrix ({agg})",
        min_max_value=min_max_value,
    )

    out_selected = outdir / f"{species}_celltype_pattern_heatmap_{agg}_with_pwm_selected_patterns.tsv"
    pd.DataFrame(selected_rows).to_csv(out_selected, sep="\t", index=False)

    print("[save]", out_png)
    print("[save]", out_selected)
    print("[done]", species)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--species", choices=["human", "macaque", "both"], default="both")
    parser.add_argument("--agg", choices=["mean", "max"], default="mean")
    parser.add_argument("--min-max-value", type=float, default=1.0)

    args = parser.parse_args()

    if args.species == "both":
        for sp in ["human", "macaque"]:
            run_species(sp, agg=args.agg, min_max_value=args.min_max_value)
    else:
        run_species(args.species, agg=args.agg, min_max_value=args.min_max_value)


if __name__ == "__main__":
    main()