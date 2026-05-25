#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from pathlib import Path
import difflib
import pandas as pd


# ============================================================
# config
# ============================================================
HAR_ID = "HARsv2_1687"

BASE_DIR = Path("/mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/pjt")
CASE_DIR = BASE_DIR / "runs" / "out" / "har1687_case"

HUMAN_WINDOWS_CSV = CASE_DIR / f"{HAR_ID}__human__top_windows.csv"
MACAQUE_WINDOWS_CSV = CASE_DIR / f"{HAR_ID}__macaque__top_windows.csv"

OUT_PAIR_CSV = CASE_DIR / f"{HAR_ID}__paired_windows.csv"
OUT_ALIGN_SUMMARY_CSV = CASE_DIR / f"{HAR_ID}__paired_alignment_summary.csv"
OUT_ALIGN_TXT = CASE_DIR / f"{HAR_ID}__paired_alignment.txt"

# 每个 model_label 最多保留多少对
TOP_PAIRS_PER_MODEL = 3


# ============================================================
# helpers
# ============================================================
def align_two_sequences(seq1: str, seq2: str) -> tuple[str, str]:
    """
    用 difflib 做简洁对齐，避免依赖 biopython
    """
    sm = difflib.SequenceMatcher(a=seq1, b=seq2)
    aln1, aln2 = [], []

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            aln1.append(seq1[i1:i2])
            aln2.append(seq2[j1:j2])
        elif tag == "replace":
            a = seq1[i1:i2]
            b = seq2[j1:j2]
            maxlen = max(len(a), len(b))
            aln1.append(a.ljust(maxlen, "-"))
            aln2.append(b.ljust(maxlen, "-"))
        elif tag == "delete":
            a = seq1[i1:i2]
            aln1.append(a)
            aln2.append("-" * len(a))
        elif tag == "insert":
            b = seq2[j1:j2]
            aln1.append("-" * len(b))
            aln2.append(b)

    return "".join(aln1), "".join(aln2)


def summarize_alignment(aln_h: str, aln_m: str):
    if len(aln_h) != len(aln_m):
        raise ValueError("Aligned sequences must have equal length")

    matches = 0
    mismatches = 0
    gaps = 0
    mismatch_positions = []
    gap_positions = []

    match_line = []

    for i, (h, m) in enumerate(zip(aln_h, aln_m)):
        if h == "-" or m == "-":
            gaps += 1
            gap_positions.append(i)
            match_line.append(" ")
        elif h == m:
            matches += 1
            match_line.append("|")
        else:
            mismatches += 1
            mismatch_positions.append(i)
            match_line.append("*")

    denom = matches + mismatches
    identity = matches / denom if denom > 0 else 0.0

    return {
        "identity": identity,
        "matches": matches,
        "mismatches": mismatches,
        "gaps": gaps,
        "mismatch_positions_aln": ",".join(map(str, mismatch_positions)),
        "gap_positions_aln": ",".join(map(str, gap_positions)),
        "match_line": "".join(match_line),
    }


def build_pairs(human_df: pd.DataFrame, macaque_df: pd.DataFrame) -> pd.DataFrame:
    """
    在同一 model_label 内，用 start_in_seq_0based 最近原则配对
    """
    pair_rows = []

    for model_label in sorted(set(human_df["model_label"]).intersection(set(macaque_df["model_label"]))):
        h_sub = human_df[human_df["model_label"] == model_label].copy()
        m_sub = macaque_df[macaque_df["model_label"] == model_label].copy()

        if h_sub.empty or m_sub.empty:
            continue

        candidates = []
        for _, hr in h_sub.iterrows():
            for _, mr in m_sub.iterrows():
                start_dist = abs(int(hr["start_in_seq_0based"]) - int(mr["start_in_seq_0based"]))
                rank_penalty = abs(int(hr["window_rank"]) - int(mr["window_rank"]))
                score = start_dist + rank_penalty * 5

                candidates.append({
                    "model_label": model_label,
                    "human_idx": hr.name,
                    "macaque_idx": mr.name,
                    "pair_score": score,
                    "start_dist": start_dist,
                    "rank_penalty": rank_penalty,
                })

        cand_df = pd.DataFrame(candidates).sort_values(
            ["pair_score", "start_dist", "rank_penalty"]
        ).reset_index(drop=True)

        used_h = set()
        used_m = set()
        kept = 0

        for _, c in cand_df.iterrows():
            h_idx = c["human_idx"]
            m_idx = c["macaque_idx"]

            if h_idx in used_h or m_idx in used_m:
                continue

            hr = h_sub.loc[h_idx]
            mr = m_sub.loc[m_idx]

            pair_rows.append({
                "har_id": HAR_ID,
                "model_label": model_label,
                "window_rank": int(hr["window_rank"]),   # 沿用 human rank
                "pair_score": float(c["pair_score"]),
                "start_dist": int(c["start_dist"]),
                "rank_penalty": int(c["rank_penalty"]),

                "species__human": hr["species"],
                "source_bed__human": hr["source_bed"],
                "region__human": f"{hr['chrom']}:{hr['genomic_start_0based']}-{hr['genomic_end_0based_exclusive']}",
                "window_score_sum_abs__human": hr["window_score_sum_abs"],
                "window_size__human": hr["window_size"],
                "start_in_seq_0based__human": hr["start_in_seq_0based"],
                "end_in_seq_0based_exclusive__human": hr["end_in_seq_0based_exclusive"],
                "chrom__human": hr["chrom"],
                "genomic_start_0based__human": hr["genomic_start_0based"],
                "genomic_end_0based_exclusive__human": hr["genomic_end_0based_exclusive"],
                "sequence__human": hr["sequence"],

                "species__macaque": mr["species"],
                "source_bed__macaque": mr["source_bed"],
                "region__macaque": f"{mr['chrom']}:{mr['genomic_start_0based']}-{mr['genomic_end_0based_exclusive']}",
                "window_score_sum_abs__macaque": mr["window_score_sum_abs"],
                "window_size__macaque": mr["window_size"],
                "start_in_seq_0based__macaque": mr["start_in_seq_0based"],
                "end_in_seq_0based_exclusive__macaque": mr["end_in_seq_0based_exclusive"],
                "chrom__macaque": mr["chrom"],
                "genomic_start_0based__macaque": mr["genomic_start_0based"],
                "genomic_end_0based_exclusive__macaque": mr["genomic_end_0based_exclusive"],
                "sequence__macaque": mr["sequence"],

                "same_window_coords_in_seq": bool(
                    int(hr["start_in_seq_0based"]) == int(mr["start_in_seq_0based"]) and
                    int(hr["end_in_seq_0based_exclusive"]) == int(mr["end_in_seq_0based_exclusive"])
                ),
                "sequence_same_exact": str(hr["sequence"]).upper() == str(mr["sequence"]).upper(),
                "pair_record_id": (
                    f"{HAR_ID}|{model_label}|human_rank{int(hr['window_rank'])}"
                    f"|human:{hr['chrom']}:{hr['genomic_start_0based']}-{hr['genomic_end_0based_exclusive']}"
                    f"|macaque:{mr['chrom']}:{mr['genomic_start_0based']}-{mr['genomic_end_0based_exclusive']}"
                ),
            })

            used_h.add(h_idx)
            used_m.add(m_idx)
            kept += 1
            if kept >= TOP_PAIRS_PER_MODEL:
                break

    out = pd.DataFrame(pair_rows)
    if not out.empty:
        out = out.sort_values(["model_label", "pair_score", "window_rank"]).reset_index(drop=True)
    return out


def write_alignment_txt(summary_df: pd.DataFrame, out_txt: Path):
    blocks = []

    for _, r in summary_df.iterrows():
        block = []
        block.append("=" * 120)
        block.append(f"HAR: {r['har_id']}")
        block.append(f"Model: {r['model_label']}")
        block.append(f"Window rank: {r['window_rank']}")
        block.append(f"Pair ID: {r['pair_record_id']}")
        block.append(f"Identity: {r['identity']:.4f}")
        block.append(f"Matches: {r['matches']}, Mismatches: {r['mismatches']}, Gaps: {r['gaps']}")
        block.append(f"Mismatch positions (alignment): {r['mismatch_positions_aln']}")
        block.append(f"Gap positions (alignment): {r['gap_positions_aln']}")
        block.append("")
        block.append(f"HUMAN   : {r['human_aln']}")
        block.append(f"          {r['match_line']}")
        block.append(f"MACAQUE : {r['macaque_aln']}")
        blocks.append("\n".join(block))

    out_txt.write_text("\n\n".join(blocks))


# ============================================================
# main
# ============================================================
def main():
    if not HUMAN_WINDOWS_CSV.exists():
        raise FileNotFoundError(f"Missing file: {HUMAN_WINDOWS_CSV}")
    if not MACAQUE_WINDOWS_CSV.exists():
        raise FileNotFoundError(f"Missing file: {MACAQUE_WINDOWS_CSV}")

    human_df = pd.read_csv(HUMAN_WINDOWS_CSV)
    macaque_df = pd.read_csv(MACAQUE_WINDOWS_CSV)

    print("Loaded human windows:", human_df.shape)
    print("Loaded macaque windows:", macaque_df.shape)

    pair_df = build_pairs(human_df, macaque_df)
    if pair_df.empty:
        raise ValueError("No pairs generated")

    pair_df.to_csv(OUT_PAIR_CSV, index=False)
    print("Saved:", OUT_PAIR_CSV)

    align_rows = []
    for _, row in pair_df.iterrows():
        seq_h = str(row["sequence__human"]).upper()
        seq_m = str(row["sequence__macaque"]).upper()

        aln_h, aln_m = align_two_sequences(seq_h, seq_m)
        stat = summarize_alignment(aln_h, aln_m)

        align_rows.append({
            "har_id": row["har_id"],
            "model_label": row["model_label"],
            "window_rank": row["window_rank"],
            "pair_record_id": row["pair_record_id"],

            "sequence__human": seq_h,
            "sequence__macaque": seq_m,

            "same_window_coords_in_seq": row["same_window_coords_in_seq"],
            "sequence_same_exact": row["sequence_same_exact"],

            "identity": stat["identity"],
            "matches": stat["matches"],
            "mismatches": stat["mismatches"],
            "gaps": stat["gaps"],
            "mismatch_positions_aln": stat["mismatch_positions_aln"],
            "gap_positions_aln": stat["gap_positions_aln"],

            "human_aln": aln_h,
            "macaque_aln": aln_m,
            "match_line": stat["match_line"],
        })

    align_df = pd.DataFrame(align_rows).sort_values(
        ["model_label", "window_rank"]
    ).reset_index(drop=True)

    align_df.to_csv(OUT_ALIGN_SUMMARY_CSV, index=False)
    print("Saved:", OUT_ALIGN_SUMMARY_CSV)

    write_alignment_txt(align_df, OUT_ALIGN_TXT)
    print("Saved:", OUT_ALIGN_TXT)

    print("Done.")


if __name__ == "__main__":
    main()