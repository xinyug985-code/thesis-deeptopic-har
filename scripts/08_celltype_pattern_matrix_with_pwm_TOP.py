#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
08_celltype_pattern_matrix_with_pwm_TOP.py

Use CREsted official function but only keep top pattern columns,
otherwise 100+ PWM logos are too compressed to see.

Run all:
    python scripts/08_celltype_pattern_matrix_with_pwm_TOP.py --top-patterns 25

Run one:
    python scripts/08_celltype_pattern_matrix_with_pwm_TOP.py --species human --agg max --top-patterns 25
"""

from pathlib import Path
import argparse
import pickle
import os

os.environ["MPLBACKEND"] = "Agg"

import matplotlib
matplotlib.use("Agg", force=True)
import crested
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
plt.switch_backend("Agg")

import pandas as pd


def patch_pdf_canvas_get_renderer():
    try:
        from matplotlib.backends.backend_pdf import FigureCanvasPdf
        from matplotlib.backends.backend_agg import RendererAgg

        if hasattr(FigureCanvasPdf, "get_renderer"):
            return

        def get_renderer(self):
            w, h = self.figure.bbox.size
            if not hasattr(self, "_renderer"):
                self._renderer = RendererAgg(int(w), int(h), self.figure.dpi)
            return self._renderer

        FigureCanvasPdf.get_renderer = get_renderer
        print("[patch] Added FigureCanvasPdf.get_renderer")
    except Exception as e:
        print("[patch failed]", type(e).__name__, e)


patch_pdf_canvas_get_renderer()


BASE = Path("/mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/pjt")
STAGE2 = BASE / "runs/out/stage2_topic_modisco_top1000"


def pat_idx(x):
    return int(str(x).replace("pattern_cluster_", ""))


def run_one(species, agg, top_patterns):
    print(f"\n===== {species} | {agg} | top {top_patterns} =====")
    print("[backend]", matplotlib.get_backend())

    pattern_pkl = STAGE2 / species / "pattern_matrix" / f"{species}_all_patterns.pkl"
    matrix_file = STAGE2 / species / "celltype_pattern_matrix" / f"{species}_celltype_by_pattern_matrix_{agg}.tsv"

    outdir = STAGE2 / species / "celltype_pattern_matrix_with_pwm_top"
    outdir.mkdir(parents=True, exist_ok=True)

    with open(pattern_pkl, "rb") as f:
        all_patterns = pickle.load(f)

    df = pd.read_csv(matrix_file, sep="\t", index_col=0)
    df = df[sorted(df.columns, key=pat_idx)]

    score = df.abs().max(axis=0).sort_values(ascending=False)
    keep_cols = list(score.head(top_patterns).index)
    keep_cols = sorted(keep_cols, key=pat_idx)
    df_top = df[keep_cols]

    print("[original shape]", df.shape)
    print("[top shape]", df_top.shape)
    print("[kept patterns]", keep_cols)

    celltypes = list(df_top.index)
    pattern_matrix = df_top.values.astype(float)

    plt.close("all")

    crested.pl.modisco.clustermap_with_pwm_logos(
        pattern_matrix,
        classes=celltypes,
        pattern_dict=all_patterns,
        width=max(30, top_patterns * 1.8),
        height=10,
        grid=True,
        dendrogram_ratio=(0.04, 0.18),
        logo_x_multiplier=2.5,
        logo_height_fraction=0.55,
        logo_y_padding=0.15,
    )

    fig = plt.gcf()
    fig.canvas.draw()

    out_png = outdir / f"{species}_celltype_by_pattern_matrix_{agg}_top{top_patterns}_with_pwm.png"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close("all")

    out_tsv = outdir / f"{species}_celltype_by_pattern_matrix_{agg}_top{top_patterns}_patterns.tsv"
    pd.DataFrame({
        "pattern_cluster": keep_cols,
        "pattern_index": [pat_idx(x) for x in keep_cols],
        "max_abs_score": [score[x] for x in keep_cols],
    }).to_csv(out_tsv, sep="\t", index=False)

    print("[saved]", out_png)
    print("[saved patterns]", out_tsv)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--species", choices=["human", "macaque", "both"], default="both")
    parser.add_argument("--agg", choices=["mean", "max", "both"], default="both")
    parser.add_argument("--top-patterns", type=int, default=25)
    args = parser.parse_args()

    species_list = ["human", "macaque"] if args.species == "both" else [args.species]
    agg_list = ["mean", "max"] if args.agg == "both" else [args.agg]

    for sp in species_list:
        for ag in agg_list:
            try:
                run_one(sp, ag, args.top_patterns)
            except Exception as e:
                print(f"[ERROR] {sp} | {ag}: {type(e).__name__}: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
