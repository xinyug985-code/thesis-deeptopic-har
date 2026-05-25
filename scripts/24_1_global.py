from __future__ import annotations
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# paths
# =========================
INPUT = Path("/mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/pjt/runs/out/har_integrated/master_har_3species_2models_wide.csv")
OUT_DIR = Path("/mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/pjt/runs/out/har_global")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# params
# =========================
HIGH_TH = 0.4     
DIFF_TH = 0.1     

# =========================
# load
# =========================
df = pd.read_csv(INPUT)
print("Loaded:", INPUT)
print("Shape:", df.shape)

df = df[df["species"] == "human"].copy()

def classify(row):
    h = row["top_score__human_model"]
    m = row["top_score__macaque_model"]

    if (h > HIGH_TH) and (h - m > DIFF_TH):
        return "human_high"
    elif (m > HIGH_TH) and (m - h > DIFF_TH):
        return "macaque_high"
    elif (h > HIGH_TH) and (m > HIGH_TH):
        return "conserved"
    else:
        return "low_or_ambiguous"

df["category"] = df.apply(classify, axis=1)

# =========================
# table
# =========================
out_csv = OUT_DIR / "har_with_category.csv"
df.to_csv(out_csv, index=False)
print("Saved:", out_csv)

# =========================
# scatter plot
# =========================
plt.figure(figsize=(6,6))

for cat, color in zip(
    ["human_high","macaque_high","conserved","low_or_ambiguous"],
    ["red","blue","green","gray"]
):
    sub = df[df["category"] == cat]
    plt.scatter(
        sub["top_score__human_model"],
        sub["top_score__macaque_model"],
        s=10,
        label=cat,
        alpha=0.6
    )

plt.xlabel("Human model score")
plt.ylabel("Macaque model score")
plt.legend()
plt.title("HAR score comparison")

plt.savefig(OUT_DIR / "scatter_human_vs_macaque.png", dpi=300)
plt.close()

# =========================
# classify
# =========================
counts = df["category"].value_counts()

plt.figure(figsize=(6,15))
counts.plot(kind="bar")
plt.ylabel("Number of HARs")
plt.title("HAR category distribution")

plt.savefig(OUT_DIR / "category_counts.png", dpi=300)
plt.close()

# =========================
# cell type（human）
# =========================
ct_counts = (
    df.groupby(["category","best_celltype_by_mean__human_model"])
    .size()
    .unstack(fill_value=0)
)

ct_counts.plot(kind="bar", stacked=True, figsize=(10,15))
plt.ylabel("Count")
plt.title("Cell type distribution (human model)")

plt.savefig(OUT_DIR / "celltype_distribution.png", dpi=300)
plt.close()

print("All done.")