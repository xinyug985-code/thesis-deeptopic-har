#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Minimal test for CREsted / matplotlib PWM logo plotting.

Purpose:
1. Check whether CREsted plotting can run in your current environment.
2. Avoid PDF output first.
3. Save PNG only.

Run:
    python scripts/00_minimal_test_crested_pwm_logo.py --species human

If this works, then backend/logo plotting is basically OK.
"""

from pathlib import Path
import argparse
import pickle
import numpy as np

# IMPORTANT:
# For HPC .py script, keep Agg.
# If you run inside Jupyter notebook, you can comment these two lines.
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


BASE = Path("/mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/pjt")
STAGE2 = BASE / "runs/out/stage2_topic_modisco_top1000"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--species", choices=["human", "macaque"], default="human")
    parser.add_argument("--pattern", default="pattern_cluster_0")
    args = parser.parse_args()

    species = args.species
    pattern_name = args.pattern

    pattern_pkl = STAGE2 / species / "pattern_matrix" / f"{species}_all_patterns.pkl"
    outdir = STAGE2 / species / "minimal_pwm_test"
    outdir.mkdir(parents=True, exist_ok=True)

    print("[load]", pattern_pkl)

    with open(pattern_pkl, "rb") as f:
        all_patterns = pickle.load(f)

    print("[type all_patterns]", type(all_patterns))

    # Get one pattern
    if isinstance(all_patterns, dict):
        print("[example keys]", list(all_patterns.keys())[:10])

        if pattern_name in all_patterns:
            pat = all_patterns[pattern_name]
        else:
            # fallback: first pattern
            first_key = list(all_patterns.keys())[0]
            print(f"[WARN] {pattern_name} not found, use first key: {first_key}")
            pat = all_patterns[first_key]
            pattern_name = str(first_key)
    else:
        print("[len all_patterns]", len(all_patterns))
        pat = all_patterns[0]
        pattern_name = "pattern_0"

    print("[selected pattern]", pattern_name)
    print("[pattern type]", type(pat))

    # Try to inspect pattern attributes
    if isinstance(pat, dict):
        print("[pattern keys]", pat.keys())
    else:
        attrs = [a for a in dir(pat) if not a.startswith("_")]
        print("[pattern attrs head]", attrs[:30])

    # Try official CREsted plotting first
    try:
        import crested.pl as cpl

        print("[try] crested.pl.plot_pwm_logo / plot_pwm_logos / plot_patterns")

        # Different CREsted versions may expose different names.
        if hasattr(cpl, "plot_pwm_logo"):
            print("[use] cpl.plot_pwm_logo")
            fig = cpl.plot_pwm_logo(pat)
        elif hasattr(cpl, "plot_pwm_logos"):
            print("[use] cpl.plot_pwm_logos")
            fig = cpl.plot_pwm_logos([pat])
        elif hasattr(cpl, "plot_patterns"):
            print("[use] cpl.plot_patterns")
            fig = cpl.plot_patterns([pat])
        else:
            raise AttributeError("No known CREsted PWM plotting function found.")

        out_png = outdir / f"{species}_{pattern_name}_official_crested_pwm_test.png"

        # Some functions return None and draw on current figure.
        if fig is None:
            fig = plt.gcf()

        fig.savefig(out_png, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print("[SUCCESS official]", out_png)
        return

    except Exception as e:
        print("[official plotting failed]")
        print(type(e).__name__, e)
        print("[fallback] Try simple PWM heatmap only.")

    # Fallback: not a logo, just verify PWM array can be extracted and saved
    pwm = None

    if isinstance(pat, dict):
        for key in ["sequence", "pwm", "ppm"]:
            if key in pat:
                pwm = np.asarray(pat[key])
                print("[use key]", key)
                break
    else:
        for attr in ["sequence", "pwm", "ppm"]:
            if hasattr(pat, attr):
                pwm = np.asarray(getattr(pat, attr))
                print("[use attr]", attr)
                break

    if pwm is None:
        pwm = np.asarray(pat)

    print("[pwm shape]", pwm.shape)

    if pwm.ndim != 2:
        raise ValueError(f"Cannot interpret pattern as 2D PWM: shape={pwm.shape}")

    fig, ax = plt.subplots(figsize=(max(4, pwm.shape[0] * 0.25), 2.2))
    im = ax.imshow(pwm.T, aspect="auto", interpolation="nearest")
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(["A", "C", "G", "T"])
    ax.set_xlabel("position")
    ax.set_title(f"{species} {pattern_name} PWM fallback heatmap")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

    out_png = outdir / f"{species}_{pattern_name}_fallback_pwm_heatmap_test.png"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("[SUCCESS fallback]", out_png)


if __name__ == "__main__":
    main()
