#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Minimal PWM logo test v3.

Adapted to your actual saved structure:

all_patterns["0"] = {
    "pattern": <object>,
    "ic": ...,
    "classes": ...,
    "instances": {
        "Topic77_neg_patterns_3": {
            "sequence": array(L, 4),
            "ppm": array(L, 4),
            "contrib_scores": ...,
            "hypothetical_contribs": ...,
            ...
        },
        ...
    }
}

This script:
1. Loads one outer pattern cluster.
2. Picks one instance inside it.
3. Extracts instance["ppm"] or instance["sequence"].
4. Draws a minimal PWM logo PNG.

Run:
    python scripts/00_minimal_test_crested_pwm_logo_v3.py --species human
    python scripts/00_minimal_test_crested_pwm_logo_v3.py --species macaque
    python scripts/00_minimal_test_crested_pwm_logo_v3.py --species human --pattern-key 0 --instance-index 0
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


def normalize_pwm(x):
    pwm = np.asarray(x, dtype=float)

    print("[raw pwm shape]", pwm.shape)

    if pwm.ndim != 2:
        raise ValueError(f"PWM must be 2D, got shape={pwm.shape}")

    if pwm.shape[0] == 4 and pwm.shape[1] != 4:
        pwm = pwm.T
        print("[transpose] 4 x L -> L x 4")

    if pwm.shape[1] != 4:
        raise ValueError(f"Expected L x 4 PWM/PPM, got shape={pwm.shape}")

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


def get_instance_pwm(outer, instance_index=0):
    if "instances" not in outer:
        raise KeyError(f"Outer pattern has no 'instances'. Keys: {outer.keys()}")

    instances = outer["instances"]
    if not isinstance(instances, dict) or len(instances) == 0:
        raise ValueError("instances is empty or not a dict")

    instance_keys = list(instances.keys())
    print("[n instances]", len(instance_keys))
    print("[example instance keys]", instance_keys[:10])

    if instance_index >= len(instance_keys):
        print(f"[WARN] instance_index {instance_index} too large, use 0")
        instance_index = 0

    inst_key = instance_keys[instance_index]
    inst = instances[inst_key]

    print("[selected instance]", inst_key)
    print("[instance keys]", list(inst.keys()))

    # Prefer ppm because it is already probability matrix.
    for key in ["ppm", "sequence"]:
        if key in inst:
            print(f"[use] instance['{key}']")
            return normalize_pwm(inst[key]), inst_key

    raise KeyError(f"No ppm/sequence in instance. Keys: {inst.keys()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--species", choices=["human", "macaque"], default="human")
    parser.add_argument("--pattern-key", default="0")
    parser.add_argument("--instance-index", type=int, default=0)
    args = parser.parse_args()

    species = args.species
    pattern_key = str(args.pattern_key)

    pattern_pkl = STAGE2 / species / "pattern_matrix" / f"{species}_all_patterns.pkl"
    outdir = STAGE2 / species / "minimal_pwm_test_v3"
    outdir.mkdir(parents=True, exist_ok=True)

    print("[load]", pattern_pkl)

    with open(pattern_pkl, "rb") as f:
        all_patterns = pickle.load(f)

    print("[type all_patterns]", type(all_patterns))

    keys = list(all_patterns.keys())
    print("[n outer patterns]", len(keys))
    print("[example outer keys]", keys[:10])

    if pattern_key not in all_patterns:
        print(f"[WARN] pattern key {pattern_key} not found, use first key {keys[0]}")
        pattern_key = str(keys[0])

    outer = all_patterns[pattern_key]
    print("[selected outer pattern]", pattern_key)
    print("[outer keys]", list(outer.keys()))

    pwm, inst_key = get_instance_pwm(outer, instance_index=args.instance_index)

    # Heatmap
    fig, ax = plt.subplots(figsize=(max(4, pwm.shape[0] * 0.25), 2.2))
    im = ax.imshow(pwm.T, aspect="auto", interpolation="nearest")
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(BASES)
    ax.set_xlabel("position")
    ax.set_title(f"{species} outer {pattern_key} | {inst_key}")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

    out_heatmap = outdir / f"{species}_outer{pattern_key}_inst{args.instance_index}_heatmap.png"
    fig.savefig(out_heatmap, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("[saved heatmap]", out_heatmap)

    # Logo
    fig, ax = plt.subplots(figsize=(max(4, pwm.shape[0] * 0.28), 2.2))
    draw_logo(ax, pwm)
    ax.set_title(f"{species} outer {pattern_key} | {inst_key}", fontsize=8)

    out_logo = outdir / f"{species}_outer{pattern_key}_inst{args.instance_index}_logo.png"
    fig.savefig(out_logo, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("[saved logo]", out_logo)

    print("[done]")


if __name__ == "__main__":
    main()
