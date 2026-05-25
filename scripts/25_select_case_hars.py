from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np


# =========================
# paths
# =========================
IN_DIR = Path("/mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/pjt/runs/out/har_integrated")
WIDE_CSV = IN_DIR / "master_har_3species_2models_wide.csv"

OUT_DIR = IN_DIR / "case_study"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_N_PER_GROUP = 10


def main():
    print("Reading:", WIDE_CSV)
    df = pd.read_csv(WIDE_CSV)
    print("Input shape:", df.shape)

    required = [
        "har_id",
        "species",
        "top_score__human_model",
        "top_score__macaque_model",
        "best_celltype_by_mean__human_model",
        "best_celltype_by_mean__macaque_model",
    ]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    df["top_score__human_model"] = pd.to_numeric(df["top_score__human_model"], errors="coerce")
    df["top_score__macaque_model"] = pd.to_numeric(df["top_score__macaque_model"], errors="coerce")

    df["delta_human_minus_macaque"] = (
        df["top_score__human_model"] - df["top_score__macaque_model"]
    )
    df["mean_top_score_2models"] = (
        df["top_score__human_model"] + df["top_score__macaque_model"]
    ) / 2.0
    df["abs_delta_2models"] = df["delta_human_minus_macaque"].abs()
    df["min_top_score_2models"] = df[
        ["top_score__human_model", "top_score__macaque_model"]
    ].min(axis=1)

    df = df.dropna(subset=["top_score__human_model", "top_score__macaque_model"]).copy()
    print("After dropna:", df.shape)

    # ========== human-specific ==========
    human_specific = df.sort_values(
        ["delta_human_minus_macaque", "top_score__human_model"],
        ascending=[False, False]
    ).head(TOP_N_PER_GROUP).copy()

    # ========== macaque-specific ==========
    macaque_specific = df.sort_values(
        ["delta_human_minus_macaque", "top_score__macaque_model"],
        ascending=[True, False]
    ).head(TOP_N_PER_GROUP).copy()

    # ========== conserved ==========
    conserved = df.sort_values(
        ["min_top_score_2models", "abs_delta_2models"],
        ascending=[False, True]
    ).head(TOP_N_PER_GROUP).copy()

    out1 = OUT_DIR / "selected_hars_human_specific.csv"
    out2 = OUT_DIR / "selected_hars_macaque_specific.csv"
    out3 = OUT_DIR / "selected_hars_conserved.csv"

    human_specific.to_csv(out1, index=False)
    macaque_specific.to_csv(out2, index=False)
    conserved.to_csv(out3, index=False)

    print("Saved:", out1)
    print("Saved:", out2)
    print("Saved:", out3)

    human_specific["group"] = "human_specific"
    macaque_specific["group"] = "macaque_specific"
    conserved["group"] = "conserved"

    all_selected = pd.concat(
        [human_specific, macaque_specific, conserved],
        axis=0,
        ignore_index=True
    ).drop_duplicates(subset=["har_id", "species"])

    out_all = OUT_DIR / "selected_hars_all_groups.csv"
    all_selected.to_csv(out_all, index=False)
    print("Saved:", out_all)

    print("\nPreview: human_specific")
    print(
        human_specific[
            [
                "har_id",
                "species",
                "top_score__human_model",
                "top_score__macaque_model",
                "delta_human_minus_macaque",
                "best_celltype_by_mean__human_model",
                "best_celltype_by_mean__macaque_model",
            ]
        ].head()
    )

    print("\nPreview: macaque_specific")
    print(
        macaque_specific[
            [
                "har_id",
                "species",
                "top_score__human_model",
                "top_score__macaque_model",
                "delta_human_minus_macaque",
                "best_celltype_by_mean__human_model",
                "best_celltype_by_mean__macaque_model",
            ]
        ].head()
    )

    print("\nPreview: conserved")
    print(
        conserved[
            [
                "har_id",
                "species",
                "top_score__human_model",
                "top_score__macaque_model",
                "delta_human_minus_macaque",
                "best_celltype_by_mean__human_model",
                "best_celltype_by_mean__macaque_model",
            ]
        ].head()
    )


if __name__ == "__main__":
    main()