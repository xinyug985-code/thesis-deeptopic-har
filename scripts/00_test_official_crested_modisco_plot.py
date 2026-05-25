#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import crested

BASE = Path("/mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/pjt")
STAGE2 = BASE / "runs/out/stage2_topic_modisco_top1000"

species = "human"

pkl_file = STAGE2 / species / "pattern_matrix" / f"{species}_all_patterns.pkl"
outdir = STAGE2 / species / "official_modisco_test"
outdir.mkdir(parents=True, exist_ok=True)

with open(pkl_file, "rb") as f:
    all_patterns = pickle.load(f)

# 1. 教程里的 selected_instances
crested.pl.modisco.selected_instances(all_patterns, [0])
plt.savefig(outdir / "selected_instances_pattern0.png", dpi=300, bbox_inches="tight")
plt.close("all")

# 2. 教程里的 class_instances
crested.pl.modisco.class_instances(all_patterns, 0)
plt.savefig(outdir / "class_instances_pattern0.png", dpi=300, bbox_inches="tight")
plt.close("all")

print("[done]", outdir)