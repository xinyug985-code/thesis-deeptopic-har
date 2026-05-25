#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

from Bio import pairwise2


# ============================================================
# paths
# ============================================================
BASE_DIR = Path("/mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/pjt")
CASE_DIR = BASE_DIR / "runs" / "out" / "case6_contribution_crossseq"

PAIRED_CSV = CASE_DIR / "selected5_human_macaque_window_pairs.csv"

OUT_CSV = CASE_DIR / "selected3_paired_window_alignment_summary.csv"
OUT_TXT = CASE_DIR / "selected3_paired_window_alignment.txt"


# ============================================================
# params
# ============================================================
KEEP_HARS = [
    "HARsv2_0920",
    "HARsv2_2699",
    "HARsv2_1547",
]

KEEP_RULES = {
    "HARsv2_0920": {
        "macaque_model": [1],
        "human_model": [3],
    },
    "HARsv2_2699": {
        "human_model": [1, 2],
        "macaque_model": [2],
    },
    "HARsv2_1547": {
        "macaque_model": [1, 2],
    },
}

# pairwise2：localms(match, mismatch, open, extend)
MATCH_SCORE = 2
MISMATCH_SCORE = -1
GAP_OPEN = -2
GAP_EXTEND = -0.5


# ============================================================
# helpers
# ============================================================
def reverse_complement(seq: str) -> str:
    comp = str.maketrans("ACGTNacgtn", "TGCANtgcan")
    return seq.translate(comp)[::-1]


def parse_keep(row: pd.Series) -> bool:
    har_id = str(row["har_id"])
    model_label = str(row["model_label"])
    rank = int(row["window_rank"])

    if har_id not in KEEP_RULES:
        return False
    if model_label not in KEEP_RULES[har_id]:
        return False
    if rank not in KEEP_RULES[har_id][model_label]:
        return False
    return True


def count_alignment_stats(aln_h: str, aln_m: str) -> dict:
    if len(aln_h) != len(aln_m):
        raise ValueError("Aligned sequences have different lengths")

    matches = 0
    mismatches = 0
    gaps = 0
    informative = 0

    mismatch_positions = []
    gap_positions = []

    for i, (a, b) in enumerate(zip(aln_h, aln_m)):
        if a == "-" or b == "-":
            gaps += 1
            gap_positions.append(i)
            continue

        informative += 1
        if a == b:
            matches += 1
        else:
            mismatches += 1
            mismatch_positions.append(i)

    identity = matches / informative if informative > 0 else np.nan

    return {
        "alignment_length": len(aln_h),
        "informative_length": informative,
        "matches": matches,
        "mismatches": mismatches,
        "gaps": gaps,
        "identity": identity,
        "mismatch_positions_aln": ",".join(map(str, mismatch_positions)),
        "gap_positions_aln": ",".join(map(str, gap_positions)),
    }


def pick_best_alignment(seq_h: str, seq_m: str) -> dict:
    seq_h = str(seq_h).upper()
    seq_m = str(seq_m).upper()
    seq_m_rc = reverse_complement(seq_m)

    alns_fwd = pairwise2.align.localms(
        seq_h, seq_m,
        MATCH_SCORE, MISMATCH_SCORE, GAP_OPEN, GAP_EXTEND,
        one_alignment_only=True
    )
    alns_rc = pairwise2.align.localms(
        seq_h, seq_m_rc,
        MATCH_SCORE, MISMATCH_SCORE, GAP_OPEN, GAP_EXTEND,
        one_alignment_only=True
    )

    best_fwd = alns_fwd[0] if len(alns_fwd) > 0 else None
    best_rc = alns_rc[0] if len(alns_rc) > 0 else None

    if best_fwd is None and best_rc is None:
        raise ValueError("No alignment found")

    if best_rc is None or (best_fwd is not None and best_fwd.score >= best_rc.score):
        aln = best_fwd
        used_orientation = "forward"
        seq_m_used = seq_m
    else:
        aln = best_rc
        used_orientation = "reverse_complement"
        seq_m_used = seq_m_rc

    aln_h = aln.seqA
    aln_m = aln.seqB
    score = float(aln.score)
    start = int(aln.start)
    end = int(aln.end)

    stats = count_alignment_stats(aln_h, aln_m)

    return {
        "human_aln": aln_h,
        "macaque_aln": aln_m,
        "alignment_score": score,
        "alignment_start": start,
        "alignment_end": end,
        "macaque_orientation_used": used_orientation,
        "macaque_sequence_used_for_alignment": seq_m_used,
        **stats,
    }


def build_marker_line(aln_h: str, aln_m: str) -> str:
    chars = []
    for a, b in zip(aln_h, aln_m):
        if a == "-" or b == "-":
            chars.append(" ")
        elif a == b:
            chars.append("|")
        else:
            chars.append("*")
    return "".join(chars)


# ============================================================
# main
# ============================================================
def main():
    if not PAIRED_CSV.exists():
        raise FileNotFoundError(f"Missing paired csv: {PAIRED_CSV}")

    df = pd.read_csv(PAIRED_CSV)
    print("Loaded paired table:", PAIRED_CSV)
    print("Shape:", df.shape)

    df = df[df["har_id"].astype(str).isin(KEEP_HARS)].copy()
    df = df[df.apply(parse_keep, axis=1)].copy()

    if df.empty:
        raise ValueError("No rows left after filtering KEEP_HARS / KEEP_RULES")

    print("Filtered rows:", df.shape[0])

    out_rows = []
    text_blocks = []

    for _, row in df.iterrows():
        har_id = str(row["har_id"])
        model_label = str(row["model_label"])
        rank = int(row["window_rank"])
        pair_id = str(row["pair_record_id"])

        seq_h = str(row["sequence__human"]).upper()
        seq_m = str(row["sequence__macaque"]).upper()

        print("=" * 80)
        print(f"Processing {pair_id}")

        res = pick_best_alignment(seq_h, seq_m)
        marker = build_marker_line(res["human_aln"], res["macaque_aln"])

        out_rows.append({
            "har_id": har_id,
            "model_label": model_label,
            "window_rank": rank,
            "pair_record_id": pair_id,

            "target_topic__human": row.get("target_topic__human", ""),
            "target_topic__macaque": row.get("target_topic__macaque", ""),
            "best_celltype__human": row.get("best_celltype__human", ""),
            "best_celltype__macaque": row.get("best_celltype__macaque", ""),

            "window_score_sum_abs__human": row.get("window_score_sum_abs__human", np.nan),
            "window_score_sum_abs__macaque": row.get("window_score_sum_abs__macaque", np.nan),
            "window_max_abs__human": row.get("window_max_abs__human", np.nan),
            "window_max_abs__macaque": row.get("window_max_abs__macaque", np.nan),

            "start_in_seq_0based__human": row.get("start_in_seq_0based__human", np.nan),
            "end_in_seq_0based_exclusive__human": row.get("end_in_seq_0based_exclusive__human", np.nan),
            "start_in_seq_0based__macaque": row.get("start_in_seq_0based__macaque", np.nan),
            "end_in_seq_0based_exclusive__macaque": row.get("end_in_seq_0based_exclusive__macaque", np.nan),

            "sequence__human": seq_h,
            "sequence__macaque": seq_m,
            "same_window_coords_in_seq": row.get("same_window_coords_in_seq", False),
            "sequence_same_exact": row.get("sequence_same_exact", False),

            **res,
        })

        text_blocks.append(
            "\n".join([
                "=" * 120,
                f"HAR: {har_id}",
                f"Model: {model_label}",
                f"Rank: {rank}",
                f"Pair ID: {pair_id}",
                f"Human topic/celltype: {row.get('target_topic__human', '')} / {row.get('best_celltype__human', '')}",
                f"Macaque topic/celltype: {row.get('target_topic__macaque', '')} / {row.get('best_celltype__macaque', '')}",
                f"Alignment score: {res['alignment_score']}",
                f"Identity: {res['identity']:.4f}",
                f"Matches: {res['matches']}, Mismatches: {res['mismatches']}, Gaps: {res['gaps']}",
                f"Macaque orientation used: {res['macaque_orientation_used']}",
                f"Mismatch positions (alignment): {res['mismatch_positions_aln']}",
                f"Gap positions (alignment): {res['gap_positions_aln']}",
                "",
                "HUMAN   : " + res["human_aln"],
                "         " + marker,
                "MACAQUE : " + res["macaque_aln"],
                "",
            ])
        )

    out_df = pd.DataFrame(out_rows)

    out_df = out_df.sort_values(
        ["har_id", "model_label", "window_rank"]
    ).reset_index(drop=True)

    out_df.to_csv(OUT_CSV, index=False)
    print("Saved:", OUT_CSV)

    with open(OUT_TXT, "w") as f:
        f.write("\n".join(text_blocks))
    print("Saved:", OUT_TXT)

    print("\nDone.")
    print("Main outputs:")
    print(OUT_CSV)
    print(OUT_TXT)


if __name__ == "__main__":
    main()