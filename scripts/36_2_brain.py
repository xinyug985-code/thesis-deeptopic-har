#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from pathlib import Path
import pandas as pd
import math

# =========================
# paths
# =========================
BASE = Path("/mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/pjt")
CASE_DIR = BASE / "runs" / "out" / "har1687_case"

HAR_ID = "HARsv2_1687"

HUMAN_WINDOW_CSV = CASE_DIR / f"{HAR_ID}__human__top_windows.csv"
MACAQUE_WINDOW_CSV = CASE_DIR / f"{HAR_ID}__macaque__top_windows.csv"


FIMO_TSV = BASE / "data" / "fimo_1687" / "fimo.tsv"

OUT_WINDOW_MERGED = CASE_DIR / f"{HAR_ID}__all_top_windows_merged.csv"
OUT_ALL = CASE_DIR / f"{HAR_ID}__fimo_annotated_all.csv"
OUT_BRAIN = CASE_DIR / f"{HAR_ID}__fimo_brain_filtered.csv"
OUT_LAYOUT = CASE_DIR / f"{HAR_ID}__fimo_layout_table.csv"

BRAIN_PATTERNS = [
    "SOX", "DLX", "NEUROD", "NEUROG", "ASCL", "OLIG",
    "FOX", "FOXP", "TCF", "MEF2", "POU", "LHX", "SPI"
]


# ============================================================
# helpers
# ============================================================
def parse_sequence_name(x: str) -> pd.Series:
    """
    Expected FIMO sequence_name format:
    HARsv2_1687|human|human_model|rank1|chr20:63102395-63102425

    Parsed fields:
    - har_id
    - seq_species
    - model_label
    - window_rank
    - region_from_seqname
    """
    parts = str(x).split("|")
    out = {
        "har_id": None,
        "seq_species": None,
        "model_label": None,
        "window_rank": None,
        "region_from_seqname": None,
    }

    if len(parts) >= 5:
        out["har_id"] = parts[0]
        out["seq_species"] = parts[1]
        out["model_label"] = parts[2]

        rank_part = parts[3]
        if str(rank_part).startswith("rank"):
            try:
                out["window_rank"] = int(str(rank_part).replace("rank", ""))
            except Exception:
                out["window_rank"] = None

        out["region_from_seqname"] = parts[4]

    return pd.Series(out)


def load_window_tables() -> pd.DataFrame:
    if not HUMAN_WINDOW_CSV.exists():
        raise FileNotFoundError(f"Missing file: {HUMAN_WINDOW_CSV}")
    if not MACAQUE_WINDOW_CSV.exists():
        raise FileNotFoundError(f"Missing file: {MACAQUE_WINDOW_CSV}")

    human = pd.read_csv(HUMAN_WINDOW_CSV)
    macaque = pd.read_csv(MACAQUE_WINDOW_CSV)

    merged = pd.concat([human, macaque], axis=0, ignore_index=True)

    # 补充一些统一字段，方便后面和之前脚本一致
    merged["har_id"] = merged["har_id"].astype(str)
    merged["seq_species"] = merged["species"].astype(str)
    merged["region"] = (
        merged["chrom"].astype(str)
        + ":"
        + merged["genomic_start_0based"].astype(str)
        + "-"
        + merged["genomic_end_0based_exclusive"].astype(str)
    )
    merged["window_region"] = merged["region"]

    return merged


# ============================================================
# main
# ============================================================
def main():
    if not FIMO_TSV.exists():
        raise FileNotFoundError(f"Missing fimo.tsv: {FIMO_TSV}")

    win = load_window_tables()
    win.to_csv(OUT_WINDOW_MERGED, index=False)
    print("Saved:", OUT_WINDOW_MERGED)

    fimo = pd.read_csv(FIMO_TSV, sep="\t", comment="#")

    print("Loaded windows:", win.shape)
    print("Loaded fimo:", fimo.shape)
    print("Window columns:", win.columns.tolist())
    print("FIMO columns:", fimo.columns.tolist())

    # -------------------------
    # parse sequence_name from FIMO
    # -------------------------
    if "sequence_name" not in fimo.columns:
        raise ValueError("FIMO file missing required column: sequence_name")

    meta = fimo["sequence_name"].apply(parse_sequence_name)
    fimo = pd.concat([fimo, meta], axis=1)

    # TF name: prefer motif_alt_id, otherwise motif_id
    if "motif_alt_id" in fimo.columns:
        fimo["tf_name"] = fimo["motif_alt_id"].fillna("")
    else:
        fimo["tf_name"] = ""
    fimo.loc[fimo["tf_name"].astype(str).str.strip() == "", "tf_name"] = fimo["motif_id"].astype(str)

    # -------------------------
    # unify merge key types
    # -------------------------
    merge_keys = ["har_id", "seq_species", "model_label", "window_rank"]

    for c in ["har_id", "seq_species", "model_label"]:
        if c not in win.columns:
            raise ValueError(f"Window CSV missing required merge key: {c}")
        fimo[c] = fimo[c].astype(str).str.strip()
        win[c] = win[c].astype(str).str.strip()

    if "window_rank" not in win.columns:
        raise ValueError("Window CSV missing required merge key: window_rank")

    fimo["window_rank"] = pd.to_numeric(fimo["window_rank"], errors="coerce")
    win["window_rank"] = pd.to_numeric(win["window_rank"], errors="coerce")

    print("\nMerge key dtypes before cleanup:")
    print("fimo:")
    print(fimo[merge_keys].dtypes)
    print("win:")
    print(win[merge_keys].dtypes)

    print("bad fimo window_rank:", fimo["window_rank"].isna().sum())
    print("bad win  window_rank:", win["window_rank"].isna().sum())

    fimo = fimo.dropna(subset=["window_rank"]).copy()
    win = win.dropna(subset=["window_rank"]).copy()

    fimo["window_rank"] = fimo["window_rank"].astype(int)
    win["window_rank"] = win["window_rank"].astype(int)

    # -------------------------
    # merge
    # -------------------------
    merged = pd.merge(
        fimo,
        win,
        on=["har_id", "seq_species", "model_label", "window_rank"],
        how="left",
        suffixes=("", "_win")
    )

    print("\nMerged shape:", merged.shape)

    if "window_score_sum_abs" in merged.columns:
        matched_n = merged["window_score_sum_abs"].notna().sum()
        unmatched_n = merged["window_score_sum_abs"].isna().sum()
        print("Matched rows:", matched_n)
        print("Unmatched rows:", unmatched_n)

    # -------------------------
    # numeric fields / layout score
    # -------------------------
    if "p-value" in merged.columns:
        merged["p_value_num"] = pd.to_numeric(merged["p-value"], errors="coerce")
        merged["neglog10_p"] = merged["p_value_num"].clip(lower=1e-300).map(
            lambda x: -math.log10(x) if pd.notna(x) else None
        )
    else:
        merged["p_value_num"] = None
        merged["neglog10_p"] = None

    if "window_score_sum_abs" in merged.columns:
        merged["layout_score"] = merged["window_score_sum_abs"].fillna(0) * merged["neglog10_p"].fillna(0)
    else:
        merged["layout_score"] = merged["neglog10_p"].fillna(0)

    merged.to_csv(OUT_ALL, index=False)
    print("Saved:", OUT_ALL)

    # =========================
    # brain TF filter
    # =========================
    pat = "|".join(BRAIN_PATTERNS)
    brain = merged[
        merged["tf_name"].astype(str).str.contains(pat, case=False, regex=True, na=False)
    ].copy()

    print("\nBrain keyword matched rows:", brain.shape[0])

    if "q-value" in brain.columns:
        brain["q_value_num"] = pd.to_numeric(brain["q-value"], errors="coerce")
        brain = brain[(brain["q_value_num"].isna()) | (brain["q_value_num"] <= 0.2)].copy()

    print("After q-value filter:", brain.shape[0])

    sort_cols = [c for c in ["har_id", "seq_species", "model_label", "window_rank", "p_value_num"] if c in brain.columns]
    if len(sort_cols) > 0:
        brain = brain.sort_values(
            sort_cols,
            ascending=[True] * len(sort_cols)
        ).reset_index(drop=True)

    brain.to_csv(OUT_BRAIN, index=False)
    print("Saved:", OUT_BRAIN)

    # =========================
    # top layout rows per HAR × seq_species × model_label
    # =========================
    if brain.shape[0] > 0:
        layout = (
            brain.sort_values(
                ["har_id", "seq_species", "model_label", "layout_score"],
                ascending=[True, True, True, False]
            )
            .groupby(["har_id", "seq_species", "model_label"], as_index=False)
            .head(5)
            .copy()
        )
    else:
        layout = brain.copy()

    keep_cols = [
        "har_id", "seq_species", "model_label",
        "region", "window_region",
        "window_rank",
        "chrom", "genomic_start_0based", "genomic_end_0based_exclusive",
        "tf_name", "motif_id", "motif_alt_id",
        "start", "stop", "strand",
        "score", "p-value", "q-value",
        "window_score_sum_abs", "window_max_abs", "window_mean_abs",
        "layout_score",
        "sequence", "matched_sequence", "region_from_seqname"
    ]
    keep_cols = [c for c in keep_cols if c in layout.columns]
    layout = layout[keep_cols].copy()

    sort_cols = [c for c in ["har_id", "seq_species", "model_label", "window_rank"] if c in layout.columns]
    if len(sort_cols) > 0 and layout.shape[0] > 0:
        layout = layout.sort_values(sort_cols).reset_index(drop=True)

    layout.to_csv(OUT_LAYOUT, index=False)
    print("Saved:", OUT_LAYOUT)

    print("\nDone.")
    print(OUT_LAYOUT)
    print(OUT_BRAIN)


if __name__ == "__main__":
    main()