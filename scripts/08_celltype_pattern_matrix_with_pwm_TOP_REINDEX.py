#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse
import pickle
import os

# 🔴 强制 backend（非常关键）
os.environ["MPLBACKEND"] = "Agg"

import matplotlib
matplotlib.use("Agg", force=True)

import crested
matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
plt.switch_backend("Agg")

import pandas as pd
import numpy as np


# ===============================
# patch（解决 pdf renderer bug）
# ===============================
def patch_pdf_canvas():
    try:
        from matplotlib.backends.backend_pdf import FigureCanvasPdf
        from matplotlib.backends.backend_agg import RendererAgg

        if hasattr(FigureCanvasPdf, "get_renderer"):
            return

        def get_renderer(self):
            w, h = self.figure.bbox.size
            return RendererAgg(int(w), int(h), self.figure.dpi)

        FigureCanvasPdf.get_renderer = get_renderer
        print("[patch] PDF renderer fixed")
    except Exception as e:
        print("[patch fail]", e)

patch_pdf_canvas()


BASE = Path("/mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/pjt")
STAGE2 = BASE / "runs/out/stage2_topic_modisco_top1000"


def pat_idx(x):
    return int(str(x).replace("pattern_cluster_", ""))


def compute_threshold(df, top_n):
    scores = df.abs().max(axis=0).sort_values(ascending=False)

    if top_n >= len(scores):
        return -np.inf

    return float(scores.iloc[top_n - 1]) - 1e-6


def run_one(species, agg, top_n):

    print(f"\n===== {species} | {agg} =====")

    pattern_pkl = STAGE2 / species / "pattern_matrix" / f"{species}_all_patterns.pkl"
    matrix_file = STAGE2 / species / "celltype_pattern_matrix" / f"{species}_celltype_by_pattern_matrix_{agg}.tsv"

    outdir = STAGE2 / species / "celltype_pattern_matrix_with_pwm_FINAL"
    outdir.mkdir(parents=True, exist_ok=True)

    print("[patterns]", pattern_pkl)
    print("[matrix]", matrix_file)

    with open(pattern_pkl, "rb") as f:
        all_patterns = pickle.load(f)

    df = pd.read_csv(matrix_file, sep="\t", index_col=0)

    # 🔴 必须排序（保证 index 对齐）
    df = df[sorted(df.columns, key=pat_idx)]

    threshold = compute_threshold(df, top_n)

    print("[threshold]", threshold)
    print("[matrix shape]", df.shape)

    celltypes = list(df.index)
    matrix = df.values.astype(float)

    plt.close("all")

    # ===============================
    # 官方函数（核心）
    # ===============================
    grid = crested.pl.modisco.clustermap_with_pwm_logos(
        matrix,
        classes=celltypes,
        pattern_dict=all_patterns,
        width=60,
        height=12,
        grid=True,
        importance_threshold=threshold,   # 🔴关键
        logo_x_multiplier=2.0,
        logo_height_fraction=0.5,
        logo_y_padding=0.2,
        pwm_or_contrib="pwm",
    )

    # 🔴 强制 draw（否则不显示）
    fig = plt.gcf()
    fig.canvas.draw()

    out_png = outdir / f"{species}_{agg}_top{top_n}_with_pwm.png"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")

    plt.close("all")

    print("[saved]", out_png)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--species", default="both", choices=["human","macaque","both"])
    parser.add_argument("--agg", default="both", choices=["mean","max","both"])
    parser.add_argument("--top-n", type=int, default=25)

    args = parser.parse_args()

    species_list = ["human","macaque"] if args.species=="both" else [args.species]
    agg_list = ["mean","max"] if args.agg=="both" else [args.agg]

    for sp in species_list:
        for ag in agg_list:
            try:
                run_one(sp, ag, args.top_n)
            except Exception as e:
                print("[ERROR]", sp, ag, e)

    print("\nDONE")


if __name__ == "__main__":
    main()