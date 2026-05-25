import os
from pathlib import Path

os.environ["KERAS_BACKEND"] = "torch"

import pandas as pd
import anndata as ad
import crested
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path("/mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/pjt")
OUTD = BASE / "runs" / "out"

H5AD = OUTD / "deeptopic_macaque" / "prediction_inspection" / "macaque_topics_with_predictions.h5ad"
DETAIL_CSV = OUTD / "deeptopic_macaque" / "prediction_inspection" / "test_predictions_detailed.csv"
FIG_DIR = OUTD / "deeptopic_macaque" / "prediction_inspection" / "region_plots"
FIG_DIR.mkdir(parents=True, exist_ok=True)

MODEL_LAYER = "pred_macaque"


def save_region_plot(adata, region, tag):
    plt.figure(figsize=(12, 5))
    crested.pl.bar.region_predictions(
        adata,
        region,
        model_names=[MODEL_LAYER],
        title=f"{tag}: {region}"
    )
    outpath = FIG_DIR / f"{tag}_{region.replace(':','_').replace('-','_')}.png"
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved:", outpath)


def main():
    adata = ad.read_h5ad(str(H5AD))
    df = pd.read_csv(DETAIL_CSV)

    print("layers:", list(adata.layers.keys()))
    print("shape:", adata.shape)
    print("detail rows:", len(df))

    # 1) top1 correct
    df_top1_ok = df[df["correct_top1"] == True]
    if len(df_top1_ok) > 0:
        region = df_top1_ok.iloc[0]["region"]
        save_region_plot(adata, region, "top1_correct")

    # 2) top1 wrong but top3 correct
    df_top3_only = df[(df["correct_top1"] == False) & (df["correct_top3"] == True)]
    if len(df_top3_only) > 0:
        region = df_top3_only.iloc[0]["region"]
        save_region_plot(adata, region, "top3_correct_only")

    # 3) top3 wrong
    df_top3_wrong = df[df["correct_top3"] == False]
    if len(df_top3_wrong) > 0:
        region = df_top3_wrong.iloc[0]["region"]
        save_region_plot(adata, region, "top3_wrong")


if __name__ == "__main__":
    main()