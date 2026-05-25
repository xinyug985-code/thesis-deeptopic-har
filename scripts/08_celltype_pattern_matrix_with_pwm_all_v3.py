#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
08_celltype_pattern_matrix_with_pwm_all_v3.py

Robust version for:
    AttributeError: 'FigureCanvasPdf' object has no attribute 'get_renderer'

Why v3:
    In your environment, matplotlib backend is still 'pdf' even after setting Agg.
    This script:
      1. forces Agg before imports,
      2. forces Agg again after importing crested,
      3. monkey-patches FigureCanvasPdf.get_renderer if backend still becomes PDF.

Run:
    python scripts/08_celltype_pattern_matrix_with_pwm_all_v3.py

Run specific:
    python scripts/08_celltype_pattern_matrix_with_pwm_all_v3.py --species human --agg max
"""

from pathlib import Path
import argparse
import pickle
import os

# Force backend as early as possible
os.environ["MPLBACKEND"] = "Agg"

import matplotlib
matplotlib.use("Agg", force=True)

# Import crested after first backend force
import crested

# Force again, because crested / site config may reset backend
matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
plt.switch_backend("Agg")

import pandas as pd


# ------------------------------------------------------------------
# Emergency patch:
# If backend is still PDF, CREsted may call fig.canvas.get_renderer(),
# but FigureCanvasPdf does not provide get_renderer().
# We add a small Agg renderer to PDF canvas to satisfy that call.
# ------------------------------------------------------------------
def patch_pdf_canvas_get_renderer():
    try:
        from matplotlib.backends.backend_pdf import FigureCanvasPdf
        from matplotlib.backends.backend_agg import RendererAgg

        if hasattr(FigureCanvasPdf, "get_renderer"):
            return

        def get_renderer(self):
            w, h = self.figure.bbox.size
            key = (w, h, self.figure.dpi)
            old_key = getattr(self, "_last_renderer_key", None)

            if old_key != key or not hasattr(self, "_renderer"):
                self._renderer = RendererAgg(
                    int(w),
                    int(h),
                    self.figure.dpi,
                )
                self._last_renderer_key = key

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


def run_one(species, agg):
    print(f"\n===== {species} | {agg} =====")
    print("[matplotlib backend before plot]", matplotlib.get_backend())

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
    df = df[sorted(df.columns, key=pat_idx)]

    celltypes = list(df.index)
    pattern_matrix = df.values.astype(float)

    print("[matrix shape]", pattern_matrix.shape)
    print("[n celltypes]", len(celltypes))
    print("[n patterns]", len(df.columns))

    plt.close("all")

    # Tutorial official function
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

    fig = plt.gcf()
    print("[matplotlib backend after plot]", matplotlib.get_backend())
    print("[figure canvas]", type(fig.canvas))

    # Try drawing explicitly
    fig.canvas.draw()

    out_png = outdir / f"{species}_celltype_by_pattern_matrix_{agg}_with_pwm.png"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
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
