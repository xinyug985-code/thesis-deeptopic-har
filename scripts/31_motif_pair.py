from __future__ import annotations
from pathlib import Path
import pandas as pd


# ============================================================
# paths
# ============================================================
BASE_DIR = Path("/mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/pjt")
CASE_DIR = BASE_DIR / "runs" / "out" / "case6_contribution_crossseq"

INPUT_CSV = CASE_DIR / "selected6_top_contribution_windows.csv"

OUT_MOTIF_CSV = CASE_DIR / "selected6_motif_input_all_windows.csv"
OUT_MOTIF_FA = CASE_DIR / "selected6_motif_input_all_windows.fa"

OUT_PAIR_CSV = CASE_DIR / "selected5_human_macaque_window_pairs.csv"
OUT_PAIR_FA = CASE_DIR / "selected5_human_macaque_window_pairs.fa"


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing input csv: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    print("Loaded:", INPUT_CSV)
    print("Shape:", df.shape)

    required = [
        "group",
        "har_id",
        "seq_species",
        "model_label",
        "region",
        "target_topic",
        "top_score",
        "best_celltype",
        "window_rank",
        "window_score_sum_abs",
        "window_max_abs",
        "window_mean_abs",
        "window_size",
        "start_in_seq_0based",
        "end_in_seq_0based_exclusive",
        "chrom",
        "genomic_start_0based",
        "genomic_end_0based_exclusive",
        "sequence",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # ========================================================
    # Part 1: motif input table for all windows
    # ========================================================
    motif_df = df.copy()

    motif_df["motif_record_id"] = motif_df.apply(
        lambda r: (
            f"{r['har_id']}|{r['seq_species']}|{r['model_label']}|"
            f"rank{r['window_rank']}|{r['chrom']}:{r['genomic_start_0based']}-{r['genomic_end_0based_exclusive']}"
        ),
        axis=1,
    )

    motif_df = motif_df.sort_values(
        ["group", "har_id", "seq_species", "model_label", "window_rank"]
    ).reset_index(drop=True)

    motif_df.to_csv(OUT_MOTIF_CSV, index=False)
    print("Saved:", OUT_MOTIF_CSV)

    with open(OUT_MOTIF_FA, "w") as f:
        for _, row in motif_df.iterrows():
            f.write(f">{row['motif_record_id']}\n{row['sequence']}\n")
    print("Saved:", OUT_MOTIF_FA)

    # ========================================================
    # Part 2: human-macaque window pairs
    # pair by har_id + model_label + window_rank
    # ========================================================
    pair_cols = [
        "group",
        "region",
        "target_topic",
        "top_score",
        "best_celltype",
        "window_score_sum_abs",
        "window_max_abs",
        "window_mean_abs",
        "window_size",
        "start_in_seq_0based",
        "end_in_seq_0based_exclusive",
        "chrom",
        "genomic_start_0based",
        "genomic_end_0based_exclusive",
        "sequence",
    ]

    human_df = df[df["seq_species"] == "human"].copy()
    macaque_df = df[df["seq_species"] == "macaque"].copy()

    human_rename = {c: f"{c}__human" for c in pair_cols}
    macaque_rename = {c: f"{c}__macaque" for c in pair_cols}

    human_df = human_df.rename(columns=human_rename)
    macaque_df = macaque_df.rename(columns=macaque_rename)

    pair_df = pd.merge(
        human_df,
        macaque_df,
        on=["har_id", "model_label", "window_rank"],
        how="inner",
    )

    pair_df["same_window_coords_in_seq"] = (
        (pair_df["start_in_seq_0based__human"] == pair_df["start_in_seq_0based__macaque"]) &
        (pair_df["end_in_seq_0based_exclusive__human"] == pair_df["end_in_seq_0based_exclusive__macaque"])
    )

    pair_df["sequence_same_exact"] = (
        pair_df["sequence__human"].astype(str) == pair_df["sequence__macaque"].astype(str)
    )

    pair_df["pair_record_id"] = pair_df.apply(
        lambda r: (
            f"{r['har_id']}|{r['model_label']}|rank{r['window_rank']}|"
            f"human:{r['chrom__human']}:{r['genomic_start_0based__human']}-{r['genomic_end_0based_exclusive__human']}|"
            f"macaque:{r['chrom__macaque']}:{r['genomic_start_0based__macaque']}-{r['genomic_end_0based_exclusive__macaque']}"
        ),
        axis=1,
    )

    keep_cols = [
        "har_id",
        "model_label",
        "window_rank",

        "group__human",
        "region__human",
        "target_topic__human",
        "top_score__human",
        "best_celltype__human",
        "window_score_sum_abs__human",
        "window_max_abs__human",
        "window_mean_abs__human",
        "window_size__human",
        "start_in_seq_0based__human",
        "end_in_seq_0based_exclusive__human",
        "chrom__human",
        "genomic_start_0based__human",
        "genomic_end_0based_exclusive__human",
        "sequence__human",

        "group__macaque",
        "region__macaque",
        "target_topic__macaque",
        "top_score__macaque",
        "best_celltype__macaque",
        "window_score_sum_abs__macaque",
        "window_max_abs__macaque",
        "window_mean_abs__macaque",
        "window_size__macaque",
        "start_in_seq_0based__macaque",
        "end_in_seq_0based_exclusive__macaque",
        "chrom__macaque",
        "genomic_start_0based__macaque",
        "genomic_end_0based_exclusive__macaque",
        "sequence__macaque",

        "same_window_coords_in_seq",
        "sequence_same_exact",
        "pair_record_id",
    ]

    pair_df = pair_df[keep_cols].sort_values(
        ["har_id", "model_label", "window_rank"]
    ).reset_index(drop=True)

    pair_df.to_csv(OUT_PAIR_CSV, index=False)
    print("Saved:", OUT_PAIR_CSV)

    with open(OUT_PAIR_FA, "w") as f:
        for _, row in pair_df.iterrows():
            f.write(f">{row['pair_record_id']}|human\n{row['sequence__human']}\n")
            f.write(f">{row['pair_record_id']}|macaque\n{row['sequence__macaque']}\n")
    print("Saved:", OUT_PAIR_FA)

    print("\nDone.")
    print("Motif input rows:", len(motif_df))
    print("Human-macaque pairs:", len(pair_df))
    if len(pair_df) > 0:
        print("HARs in pair table:", sorted(pair_df["har_id"].unique().tolist()))


if __name__ == "__main__":
    main()