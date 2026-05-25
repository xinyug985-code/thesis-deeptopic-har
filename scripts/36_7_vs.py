#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# paths
# ============================================================
BASE_DIR = Path("/mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/pjt")
CASE_DIR = BASE_DIR / "runs" / "out" / "har1687_case"

HAR_ID = "HARsv2_1687"

ALIGN_CSV = CASE_DIR / f"{HAR_ID}__paired_alignment_summary.csv"
SENSITIVE_CSV = CASE_DIR / "36_3_human_focus_ism" / "36_top_sensitive_sites_high_only.csv"

OUT_POS_CSV = CASE_DIR / f"{HAR_ID}__sensitive_vs_mutation_positions.csv"
OUT_SUMMARY_CSV = CASE_DIR / f"{HAR_ID}__sensitive_vs_mutation_summary.csv"


# ============================================================
# params
# ============================================================
MIN_MAX_DROP = 0.02


# ============================================================
# helpers
# ============================================================
def parse_alignment_positions(aln_h: str, aln_m: str):
    """
    - human raw pos -> alignment pos
    - macaque raw pos -> alignment pos
    - alignment pos -> human raw pos / macaque raw pos
    """
    if len(aln_h) != len(aln_m):
        raise ValueError("Aligned sequences must have equal length")

    human_raw_to_aln = {}
    macaque_raw_to_aln = {}
    aln_to_human_raw = {}
    aln_to_macaque_raw = {}

    h_raw = 0
    m_raw = 0

    for aln_i, (h, m) in enumerate(zip(aln_h, aln_m)):
        if h != "-":
            human_raw_to_aln[h_raw] = aln_i
            aln_to_human_raw[aln_i] = h_raw
            h_raw += 1
        else:
            aln_to_human_raw[aln_i] = None

        if m != "-":
            macaque_raw_to_aln[m_raw] = aln_i
            aln_to_macaque_raw[aln_i] = m_raw
            m_raw += 1
        else:
            aln_to_macaque_raw[aln_i] = None

    return {
        "human_raw_to_aln": human_raw_to_aln,
        "macaque_raw_to_aln": macaque_raw_to_aln,
        "aln_to_human_raw": aln_to_human_raw,
        "aln_to_macaque_raw": aln_to_macaque_raw,
    }


def get_mismatch_aln_positions(aln_h: str, aln_m: str):
    out = []
    for i, (h, m) in enumerate(zip(aln_h, aln_m)):
        if h == "-" or m == "-":
            continue
        if h != m:
            out.append(i)
    return out


def get_gap_aln_positions(aln_h: str, aln_m: str):
    out = []
    for i, (h, m) in enumerate(zip(aln_h, aln_m)):
        if h == "-" or m == "-":
            out.append(i)
    return out


def build_sensitive_lookup(sensitive_df: pd.DataFrame):
    """
    dict，key:
      (har_id, model_label, window_rank, species_label, position_in_window_0based)
    """
    lookup = {}
    for _, r in sensitive_df.iterrows():
        key = (
            str(r["har_id"]),
            str(r["model_label"]),
            int(r["window_rank"]),
            str(r["species_label"]),
            int(r["position_in_window_0based"]),
        )
        lookup[key] = {
            "sensitive_rank": int(r["sensitive_rank"]),
            "ref_base": str(r["ref_base"]),
            "max_score_drop": float(r["max_score_drop"]),
            "mean_score_drop": float(r["mean_score_drop"]),
            "worst_alt_base": str(r["worst_alt_base"]),
            "worst_alt_score_drop": float(r["worst_alt_score_drop"]),
        }
    return lookup


# ============================================================
# main
# ============================================================
def main():
    if not ALIGN_CSV.exists():
        raise FileNotFoundError(f"Missing alignment csv: {ALIGN_CSV}")
    if not SENSITIVE_CSV.exists():
        raise FileNotFoundError(f"Missing sensitive csv: {SENSITIVE_CSV}")

    align_df = pd.read_csv(ALIGN_CSV)
    sensitive_df = pd.read_csv(SENSITIVE_CSV)

    print("Loaded alignment:", ALIGN_CSV, align_df.shape)
    print("Loaded sensitive:", SENSITIVE_CSV, sensitive_df.shape)

    # 可选：保险再筛一次
    sensitive_df = sensitive_df.loc[sensitive_df["max_score_drop"] >= MIN_MAX_DROP].copy()
    print("Sensitive rows after threshold:", sensitive_df.shape[0])

    sensitive_lookup = build_sensitive_lookup(sensitive_df)

    pos_rows = []
    summary_rows = []

    for _, row in align_df.iterrows():
        har_id = str(row["har_id"])
        model_label = str(row["model_label"])
        window_rank = int(row["window_rank"])
        pair_id = str(row["pair_record_id"])

        seq_h = str(row["sequence__human"])
        seq_m = str(row["sequence__macaque"])
        aln_h = str(row["human_aln"])
        aln_m = str(row["macaque_aln"])

        maps = parse_alignment_positions(aln_h, aln_m)
        mismatch_aln_positions = set(get_mismatch_aln_positions(aln_h, aln_m))
        gap_aln_positions = set(get_gap_aln_positions(aln_h, aln_m))

        all_aln_positions = sorted(
            set(maps["aln_to_human_raw"].keys()) | set(maps["aln_to_macaque_raw"].keys())
        )

        for aln_pos in all_aln_positions:
            h_pos = maps["aln_to_human_raw"].get(aln_pos, None)
            m_pos = maps["aln_to_macaque_raw"].get(aln_pos, None)

            h_base = aln_h[aln_pos]
            m_base = aln_m[aln_pos]

            is_gap = aln_pos in gap_aln_positions
            is_mismatch = aln_pos in mismatch_aln_positions

            # human side sensitive?
            h_sensitive = False
            h_sensitive_info = {}
            if h_pos is not None:
                key_h = (har_id, model_label, window_rank, "human", h_pos)
                if key_h in sensitive_lookup:
                    h_sensitive = True
                    h_sensitive_info = sensitive_lookup[key_h]

            # macaque side sensitive?
            m_sensitive = False
            m_sensitive_info = {}
            if m_pos is not None:
                key_m = (har_id, model_label, window_rank, "macaque", m_pos)
                if key_m in sensitive_lookup:
                    m_sensitive = True
                    m_sensitive_info = sensitive_lookup[key_m]

            overlap_sensitive_mutation = is_mismatch and (h_sensitive or m_sensitive)

            pos_rows.append({
                "har_id": har_id,
                "model_label": model_label,
                "window_rank": window_rank,
                "pair_record_id": pair_id,

                "alignment_pos_0based": aln_pos,

                "human_pos_in_window_0based": h_pos,
                "human_base": h_base,
                "human_sensitive": h_sensitive,
                "human_sensitive_rank": h_sensitive_info.get("sensitive_rank", np.nan),
                "human_max_score_drop": h_sensitive_info.get("max_score_drop", np.nan),
                "human_worst_alt_base": h_sensitive_info.get("worst_alt_base", ""),
                "human_worst_alt_score_drop": h_sensitive_info.get("worst_alt_score_drop", np.nan),

                "macaque_pos_in_window_0based": m_pos,
                "macaque_base": m_base,
                "macaque_sensitive": m_sensitive,
                "macaque_sensitive_rank": m_sensitive_info.get("sensitive_rank", np.nan),
                "macaque_max_score_drop": m_sensitive_info.get("max_score_drop", np.nan),
                "macaque_worst_alt_base": m_sensitive_info.get("worst_alt_base", ""),
                "macaque_worst_alt_score_drop": m_sensitive_info.get("worst_alt_score_drop", np.nan),

                "is_gap": is_gap,
                "is_mismatch": is_mismatch,
                "overlap_sensitive_mutation": overlap_sensitive_mutation,
            })

        pair_pos_df = pd.DataFrame([r for r in pos_rows if r["pair_record_id"] == pair_id])
        overlap_rows = pair_pos_df.loc[pair_pos_df["overlap_sensitive_mutation"]].copy()

        human_overlap_sites = overlap_rows["human_pos_in_window_0based"].dropna().astype(int).tolist()
        macaque_overlap_sites = overlap_rows["macaque_pos_in_window_0based"].dropna().astype(int).tolist()

        summary_rows.append({
            "har_id": har_id,
            "model_label": model_label,
            "window_rank": window_rank,
            "pair_record_id": pair_id,

            "sequence_same_exact": row.get("sequence_same_exact", False),
            "same_window_coords_in_seq": row.get("same_window_coords_in_seq", False),
            "identity": row.get("identity", np.nan),
            "matches": row.get("matches", np.nan),
            "mismatches": row.get("mismatches", np.nan),
            "gaps": row.get("gaps", np.nan),

            "n_sensitive_human_sites": int(pair_pos_df["human_sensitive"].sum()),
            "n_sensitive_macaque_sites": int(pair_pos_df["macaque_sensitive"].sum()),
            "n_mismatch_sites": int(pair_pos_df["is_mismatch"].sum()),
            "n_sensitive_mutation_overlap_sites": int(pair_pos_df["overlap_sensitive_mutation"].sum()),

            "human_overlap_positions_in_window": ",".join(map(str, human_overlap_sites)),
            "macaque_overlap_positions_in_window": ",".join(map(str, macaque_overlap_sites)),
        })

    pos_df = pd.DataFrame(pos_rows)
    summary_df = pd.DataFrame(summary_rows)

    pos_df = pos_df.sort_values(
        ["har_id", "model_label", "window_rank", "alignment_pos_0based"]
    ).reset_index(drop=True)

    summary_df = summary_df.sort_values(
        ["har_id", "model_label", "window_rank"]
    ).reset_index(drop=True)

    pos_df.to_csv(OUT_POS_CSV, index=False)
    summary_df.to_csv(OUT_SUMMARY_CSV, index=False)

    print("Saved:", OUT_POS_CSV)
    print("Saved:", OUT_SUMMARY_CSV)
    print("Done.")


if __name__ == "__main__":
    main()