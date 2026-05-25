#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Minimal PWM logo test v2.

This version is adapted to your actual pkl structure:

all_patterns: dict
all_patterns["0"]: {
    "pattern": array-like PWM/PPM,
    "ic": ...,
    "classes": ...,
    "instances": ...
}

Run:
    python scripts/00_minimal_test_crested_pwm_logo_v2.py --species human
    python scripts/00_minimal_test_crested_pwm_logo_v2.py --species macaque

Output:
    runs/out/stage2_topic_modisco_top1000/<species>/minimal_pwm_test_v2/
"""

from pathlib import Path
import argparse
import pickle
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib.textpath import TextPath
from matplotlib.patches import PathPatch
from matplotlib.transforms import Affine2D
from matplotlib.font_manager import FontProperties


BASE = Path("/mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/pjt")
STAGE2 = BASE / "runs/out/stage2_topic_modisco_top1000"

BASES = ["A", "C", "G", "T"]
BASE_COLORS = {
    "A": "#2ca02c",
    "C": "#1f77b4",
    "G": "#ff7f0e",
    "T": "#d62728",
}


def extract_pwm(pat):
    """
    Extract PWM/PPM matrix from one pattern object.
    Your observed structure:
        pat is dict with key "pattern"
    """
    if isinstance(pat, dict):
        print("[pattern keys]", list(pat.keys()))

        if "pattern" in pat:
            pwm = np.asarray(pat["pattern"])
            print("[use] pat['pattern']")
        elif "sequence" in pat:
            pwm = np.asarray(pat["sequence"])
            print("[use] pat['sequence']")
        elif "pwm" in pat:
            pwm = np.asarray(pat["pwm"])
            print("[use] pat['pwm']")
        elif "ppm" in pat:
            pwm = np.asarray(pat["ppm"])
            print("[use] pat['ppm']")
        else:
            raise KeyError(f"Cannot find PWM-like key in pattern dict: {pat.keys()}")

    else:
        for attr in ["pattern", "sequence", "pwm", "ppm"]:
            if hasattr(pat, attr):
                pwm = np.asarray(getattr(pat, attr))
                print(f"[use] pat.{attr}")
                break
        else:
            pwm = np.asarray(pat)
            print("[use] np.asarray(pat)")

    print("[raw pwm shape]", pwm.shape)
    print("[raw pwm dtype]", pwm.dtype)

    if pwm.ndim != 2:
        raise ValueError(f"PWM must be 2D, got shape={pwm.shape}")

    # If shape is 4 x L, transpose to L x 4
    if pwm.shape[0] == 4 and pwm.shape[1] != 4:
        pwm = pwm.T
        print("[transpose] 4 x L -> L x 4")

    if pwm.shape[1] != 4:
        raise ValueError(f"Expected L x 4 PWM, got shape={pwm.shape}")

    pwm = np.asarray(pwm, dtype=float)

    # Convert possible contribution-like values to non-negative PPM-like matrix
    # If negative values exist, shift each row to be >= 0.
    if np.nanmin(pwm) < 0:
        pwm = pwm - np.nanmin(pwm, axis=1, keepdims=True)

    pwm = np.nan_to_num(pwm, nan=0.0, posinf=0.0, neginf=0.0)
    pwm = np.clip(pwm, 0, None)

    row_sum = pwm.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    pwm = pwm / row_sum

    print("[final pwm shape]", pwm.shape)
    print("[row sum min/max]", pwm.sum(axis=1).min(), pwm.sum(axis=1).max())

    return pwm


def information_content(pwm):
    eps = 1e-8
    entropy = -np.sum(pwm * np.log2(pwm + eps), axis=1)
    ic = 2.0 - entropy
    return np.clip(ic, 0, 2)


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

            trans = (
                Affine2D()
                .scale(sx, sy)
                .translate(x + 0.05, y)
            )

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
    ax.set_xlabel("position")
    ax.set_ylabel("bits")

    for spine in ax.spines.values():
        spine.set_visible(False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--species", choices=["human", "macaque"], default="human")
    parser.add_argument("--pattern-key", default="0", help="Example: 0, 1, 2 ...")
    args = parser.parse_args()

    species = args.species
    pattern_key = str(args.pattern_key)

    pattern_pkl = STAGE2 / species / "pattern_matrix" / f"{species}_all_patterns.pkl"
    outdir = STAGE2 / species / "minimal_pwm_test_v2"
    outdir.mkdir(parents=True, exist_ok=True)

    print("[load]", pattern_pkl)

    with open(pattern_pkl, "rb") as f:
        all_patterns = pickle.load(f)

    print("[type all_patterns]", type(all_patterns))

    if isinstance(all_patterns, dict):
        keys = list(all_patterns.keys())
        print("[n patterns]", len(keys))
        print("[example keys]", keys[:10])

        if pattern_key not in all_patterns:
            print(f"[WARN] key {pattern_key} not found, use first key {keys[0]}")
            pattern_key = str(keys[0])

        pat = all_patterns[pattern_key]
    else:
        print("[n patterns]", len(all_patterns))
        idx = int(pattern_key)
        pat = all_patterns[idx]
        pattern_key = str(idx)

    print("[selected pattern]", pattern_key)
    print("[pattern type]", type(pat))

    pwm = extract_pwm(pat)

    # 1) save heatmap
    fig, ax = plt.subplots(figsize=(max(4, pwm.shape[0] * 0.25), 2.2))
    im = ax.imshow(pwm.T, aspect="auto", interpolation="nearest")
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(BASES)
    ax.set_xlabel("position")
    ax.set_title(f"{species} pattern {pattern_key} PWM heatmap")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

    out_heatmap = outdir / f"{species}_pattern_{pattern_key}_pwm_heatmap.png"
    fig.savefig(out_heatmap, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("[saved heatmap]", out_heatmap)

    # 2) save logo
    fig, ax = plt.subplots(figsize=(max(4, pwm.shape[0] * 0.28), 2.2))
    draw_logo(ax, pwm)
    ax.set_title(f"{species} pattern {pattern_key} PWM logo")

    out_logo = outdir / f"{species}_pattern_{pattern_key}_pwm_logo.png"
    fig.savefig(out_logo, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("[saved logo]", out_logo)

    print("[done]")


if __name__ == "__main__":
    main()
