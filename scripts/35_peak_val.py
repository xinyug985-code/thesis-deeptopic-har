#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import os
os.environ["KERAS_BACKEND"] = "torch"

from pathlib import Path
import gc
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import keras
import crested
import torch


# ============================================================
# paths
# ============================================================
BASE_DIR = Path("/mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/pjt")
CASE_DIR = BASE_DIR / "runs" / "out" / "case6_contribution_crossseq"

PAIRED_CSV = CASE_DIR / "selected5_human_macaque_window_pairs.csv"
ALIGN_CSV = CASE_DIR / "selected3_paired_window_alignment_summary.csv"

# 全哥 peak regression 模型
HUMAN_PEAK_MODEL_PATH = Path(
    "/mimer/NOBACKUP/groups/naiss2025-22-612/yiquan/atac/developing_brain_adjust/basemodel/checkpoints/29.keras"
)
MACAQUE_PEAK_MODEL_PATH = Path(
    "/mimer/NOBACKUP/groups/naiss2025-22-612/yiquan/macaque/macaque_brain_swift/basemodel/checkpoints/44.keras"
)

HUMAN_FA = Path("/mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/genomes/hg38/hg38.fa")
MACAQUE_FA = Path("/mimer/NOBACKUP/groups/naiss2025-22-612/yiquan/macaque/rheMac10.fa")

OUT_DIR = BASE_DIR / "runs" / "out" / "peak_model_validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# params
# ============================================================
KEEP_WINDOWS = [
    ("HARsv2_0920", "macaque_model", 1),
    ("HARsv2_0920", "human_model", 3),
    ("HARsv2_2699", "macaque_model", 2),
    ("HARsv2_2699", "human_model", 2),
]

ADD_HAR_KEYWORDS = ["HAR1A", "HAR1B"]

FIG_DPI = 300

# 关键：peak model 输入长度
PEAK_INPUT_LEN = 2114
CENTER_REGION_LEN = 500
FLANK_LEN = (PEAK_INPUT_LEN - CENTER_REGION_LEN) // 2  # 807


# ============================================================
# helpers
# ============================================================
def sanitize_name(x: str) -> str:
    x = str(x)
    for ch in [" ", "/", "\\", "|", ":", ";", ",", "(", ")", "[", "]"]:
        x = x.replace(ch, "_")
    return x


def parse_region(region: str):
    chrom, pos = region.split(":")
    start, end = pos.split("-")
    return chrom, int(start), int(end)


def get_genome(species_label: str):
    species_label = str(species_label).lower()
    if species_label == "human":
        return crested.Genome(HUMAN_FA)
    elif species_label == "macaque":
        return crested.Genome(MACAQUE_FA)
    else:
        raise ValueError(f"Unsupported species_label: {species_label}")


def fetch_sequence(species_label: str, chrom: str, start: int, end: int) -> str:
    genome = get_genome(species_label)
    seq = genome.fetch(chrom, int(start), int(end)).upper()
    return seq


def fetch_2114_sequence_from_500_region(species_label: str, region_500bp: str) -> tuple[str, dict]:
    """
    从 500bp region 构造以其为中心的 2114bp 输入序列
    返回:
      - seq_2114
      - meta
    """
    chrom, start500, end500 = parse_region(region_500bp)
    region_len = end500 - start500
    if region_len != CENTER_REGION_LEN:
        warnings.warn(f"Expected 500bp region, got {region_len} bp: {region_500bp}")

    start2114 = start500 - FLANK_LEN
    end2114 = end500 + FLANK_LEN

    if start2114 < 0:
        raise ValueError(f"2114 window start < 0 for {species_label} {region_500bp}")

    seq_2114 = fetch_sequence(species_label, chrom, start2114, end2114)
    expected_len = end2114 - start2114
    if len(seq_2114) != expected_len:
        warnings.warn(
            f"Fetched 2114 seq len {len(seq_2114)} != expected {expected_len} "
            f"for {species_label} {chrom}:{start2114}-{end2114}"
        )

    meta = {
        "chrom": chrom,
        "region500_start": start500,
        "region500_end": end500,
        "region2114_start": start2114,
        "region2114_end": end2114,
        "region500_offset_in_2114": FLANK_LEN,
    }
    return seq_2114, meta


def mutate_positions(seq: str, replacements: dict[int, str]) -> str:
    s = list(seq)
    for pos, base in replacements.items():
        s[pos] = base
    return "".join(s)


def load_peak_models():
    print("Loading peak models...")
    human_peak_model = keras.models.load_model(str(HUMAN_PEAK_MODEL_PATH), compile=False)
    macaque_peak_model = keras.models.load_model(str(MACAQUE_PEAK_MODEL_PATH), compile=False)
    print("Human peak model loaded:", HUMAN_PEAK_MODEL_PATH)
    print("Macaque peak model loaded:", MACAQUE_PEAK_MODEL_PATH)
    return human_peak_model, macaque_peak_model


def predict_with_peak_model(sequence: str, model) -> np.ndarray:
    """
    旧版 crested 风格：直接喂 string，但这里必须是 2114bp
    """
    pred = crested.tl.predict(sequence, model=model)
    pred = np.asarray(pred)

    if pred.ndim == 2 and pred.shape[0] == 1:
        pred = pred[0]
    elif pred.ndim > 2:
        pred = np.squeeze(pred)

    if pred.ndim != 1:
        raise ValueError(f"Unexpected prediction shape: {pred.shape}")

    return pred


def summarize_prediction_vector(pred: np.ndarray) -> dict:
    pred = np.asarray(pred, dtype=float)
    return {
        "pred_mean": float(np.mean(pred)),
        "pred_max": float(np.max(pred)),
        "pred_sum": float(np.sum(pred)),
        "pred_argmax": int(np.argmax(pred)),
        "pred_l2": float(np.linalg.norm(pred)),
    }


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return np.nan
    return float(np.dot(a, b) / (na * nb))


def extract_window_mismatch_positions(row: pd.Series) -> tuple[list[int], list[int]]:
    """
    从局部 window alignment 中恢复 mismatch 的 raw window 坐标
    返回:
      - human window raw mismatch positions
      - macaque window raw mismatch positions
    """
    aln_h = str(row["human_aln"])
    aln_m = str(row["macaque_aln"])

    h_raw = 0
    m_raw = 0
    h_positions = []
    m_positions = []

    for h, m in zip(aln_h, aln_m):
        h_pos = None
        m_pos = None

        if h != "-":
            h_pos = h_raw
            h_raw += 1
        if m != "-":
            m_pos = m_raw
            m_raw += 1

        if h != "-" and m != "-" and h != m:
            h_positions.append(h_pos)
            m_positions.append(m_pos)

    return h_positions, m_positions


def add_har1_rows_if_present(df: pd.DataFrame) -> pd.DataFrame:
    keep_idx = []
    for idx, row in df.iterrows():
        har_id = str(row["har_id"])
        if any(k.lower() in har_id.lower() for k in ADD_HAR_KEYWORDS):
            keep_idx.append(idx)
    if len(keep_idx) == 0:
        return df
    extra = df.loc[keep_idx].copy()
    return pd.concat([df, extra], axis=0).drop_duplicates().reset_index(drop=True)


def validate_window_matches_center500(
    seq_2114: str,
    window_seq: str,
    start_in_500: int,
    end_in_500: int,
    species_label: str,
    pair_id: str,
):
    """
    检查 paired table 的 30bp window 是否真的对应 2114 中央500bp里的那一段
    """
    region500_offset = FLANK_LEN
    sub = seq_2114[region500_offset + start_in_500 : region500_offset + end_in_500].upper()
    if sub != str(window_seq).upper():
        raise ValueError(
            f"Window sequence does not match central 500bp for {species_label} | {pair_id}\n"
            f"Expected from 2114 center: {sub}\n"
            f"Window seq in table      : {window_seq}"
        )


def run_single_case(row: pd.Series, human_peak_model, macaque_peak_model):
    har_id = str(row["har_id"])
    model_label = str(row["model_label"])
    window_rank = int(row["window_rank"])
    pair_id = str(row["pair_record_id"])

    print("\n" + "=" * 90)
    print(f"Processing {pair_id}")

    # 取 2114 输入序列（central 500 = 当前 paired region）
    region_h = str(row["region__human"])
    region_m = str(row["region__macaque"])

    full_h_2114, meta_h = fetch_2114_sequence_from_500_region("human", region_h)
    full_m_2114, meta_m = fetch_2114_sequence_from_500_region("macaque", region_m)

    # 30bp window 在 central 500 里的坐标
    h_s_500 = int(row["start_in_seq_0based__human"])
    h_e_500 = int(row["end_in_seq_0based_exclusive__human"])
    m_s_500 = int(row["start_in_seq_0based__macaque"])
    m_e_500 = int(row["end_in_seq_0based_exclusive__macaque"])

    validate_window_matches_center500(full_h_2114, row["sequence__human"], h_s_500, h_e_500, "human", pair_id)
    validate_window_matches_center500(full_m_2114, row["sequence__macaque"], m_s_500, m_e_500, "macaque", pair_id)

    # 转成 2114 坐标
    h_s_2114 = FLANK_LEN + h_s_500
    h_e_2114 = FLANK_LEN + h_e_500
    m_s_2114 = FLANK_LEN + m_s_500
    m_e_2114 = FLANK_LEN + m_e_500

    window_h = full_h_2114[h_s_2114:h_e_2114]
    window_m = full_m_2114[m_s_2114:m_e_2114]

    # alignment mismatch 位点（30bp window raw 坐标）
    h_mm_pos, m_mm_pos = extract_window_mismatch_positions(row)

    # 构造互换突变：只替换 mismatch 位点
    h_replacements = {}
    m_replacements = {}

    for hp, mp in zip(h_mm_pos, m_mm_pos):
        if hp is None or mp is None:
            continue
        if hp >= len(window_h) or mp >= len(window_m):
            continue

        h_replacements[h_s_2114 + hp] = window_m[mp]
        m_replacements[m_s_2114 + mp] = window_h[hp]

    full_h_mut = mutate_positions(full_h_2114, h_replacements) if len(h_replacements) > 0 else full_h_2114
    full_m_mut = mutate_positions(full_m_2114, m_replacements) if len(m_replacements) > 0 else full_m_2114

    # 用两个 peak model 都看一遍
    pred_h_on_hpeak = predict_with_peak_model(full_h_2114, human_peak_model)
    pred_hmut_on_hpeak = predict_with_peak_model(full_h_mut, human_peak_model)
    pred_m_on_hpeak = predict_with_peak_model(full_m_2114, human_peak_model)
    pred_mmut_on_hpeak = predict_with_peak_model(full_m_mut, human_peak_model)

    pred_h_on_mpeak = predict_with_peak_model(full_h_2114, macaque_peak_model)
    pred_hmut_on_mpeak = predict_with_peak_model(full_h_mut, macaque_peak_model)
    pred_m_on_mpeak = predict_with_peak_model(full_m_2114, macaque_peak_model)
    pred_mmut_on_mpeak = predict_with_peak_model(full_m_mut, macaque_peak_model)

    rows = []

    def add_row(model_name, seq_name, pred_ref, pred_mut):
        s_ref = summarize_prediction_vector(pred_ref)
        s_mut = summarize_prediction_vector(pred_mut)
        rows.append({
            "har_id": har_id,
            "model_label": model_label,
            "window_rank": window_rank,
            "pair_record_id": pair_id,
            "peak_model_used": model_name,
            "sequence_name": seq_name,

            "n_window_mismatches_used": len(h_replacements) if seq_name == "human" else len(m_replacements),
            "window_mismatch_positions_human": ",".join(map(str, h_mm_pos)),
            "window_mismatch_positions_macaque": ",".join(map(str, m_mm_pos)),

            "pred_mean_ref": s_ref["pred_mean"],
            "pred_max_ref": s_ref["pred_max"],
            "pred_sum_ref": s_ref["pred_sum"],
            "pred_argmax_ref": s_ref["pred_argmax"],
            "pred_l2_ref": s_ref["pred_l2"],

            "pred_mean_mut": s_mut["pred_mean"],
            "pred_max_mut": s_mut["pred_max"],
            "pred_sum_mut": s_mut["pred_sum"],
            "pred_argmax_mut": s_mut["pred_argmax"],
            "pred_l2_mut": s_mut["pred_l2"],

            "delta_mean": s_mut["pred_mean"] - s_ref["pred_mean"],
            "delta_max": s_mut["pred_max"] - s_ref["pred_max"],
            "delta_sum": s_mut["pred_sum"] - s_ref["pred_sum"],
            "delta_l2": s_mut["pred_l2"] - s_ref["pred_l2"],

            "cosine_ref_vs_mut": cosine_similarity(pred_ref, pred_mut),
            "l2_distance_ref_vs_mut": float(np.linalg.norm(pred_ref - pred_mut)),
        })

    add_row("human_peak_model", "human", pred_h_on_hpeak, pred_hmut_on_hpeak)
    add_row("human_peak_model", "macaque", pred_m_on_hpeak, pred_mmut_on_hpeak)
    add_row("macaque_peak_model", "human", pred_h_on_mpeak, pred_hmut_on_mpeak)
    add_row("macaque_peak_model", "macaque", pred_m_on_mpeak, pred_mmut_on_mpeak)

    # 保存详细向量
    vector_rows = []
    vector_sets = {
        ("human_peak_model", "human", "ref"): pred_h_on_hpeak,
        ("human_peak_model", "human", "mut"): pred_hmut_on_hpeak,
        ("human_peak_model", "macaque", "ref"): pred_m_on_hpeak,
        ("human_peak_model", "macaque", "mut"): pred_mmut_on_hpeak,
        ("macaque_peak_model", "human", "ref"): pred_h_on_mpeak,
        ("macaque_peak_model", "human", "mut"): pred_hmut_on_mpeak,
        ("macaque_peak_model", "macaque", "ref"): pred_m_on_mpeak,
        ("macaque_peak_model", "macaque", "mut"): pred_mmut_on_mpeak,
    }

    for (peak_model_used, sequence_name, state), vec in vector_sets.items():
        for i, v in enumerate(vec):
            vector_rows.append({
                "har_id": har_id,
                "model_label": model_label,
                "window_rank": window_rank,
                "pair_record_id": pair_id,
                "peak_model_used": peak_model_used,
                "sequence_name": sequence_name,
                "state": state,
                "output_index": i,
                "value": float(v),
            })

    prefix = OUT_DIR / f"{sanitize_name(har_id)}__{model_label}__rank{window_rank}"

    vec_df = pd.DataFrame(vector_rows)
    vec_csv = str(prefix) + "__prediction_vectors.csv"
    vec_df.to_csv(vec_csv, index=False)
    print("Saved:", vec_csv)

    summary_df = pd.DataFrame(rows)
    sum_csv = str(prefix) + "__peak_validation_summary.csv"
    summary_df.to_csv(sum_csv, index=False)
    print("Saved:", sum_csv)

    # plot 1
    plt.figure(figsize=(8, 4))
    sub = summary_df[summary_df["peak_model_used"] == "human_peak_model"].copy()
    x = np.arange(sub.shape[0])
    plt.bar(x - 0.15, sub["pred_l2_ref"], width=0.3, label="ref")
    plt.bar(x + 0.15, sub["pred_l2_mut"], width=0.3, label="mut")
    plt.xticks(x, sub["sequence_name"])
    plt.ylabel("Prediction vector L2")
    plt.title(f"{har_id} | {model_label} | human peak model")
    plt.legend(frameon=False)
    plt.tight_layout()
    out_png1 = str(prefix) + "__human_peak_model.png"
    out_pdf1 = str(prefix) + "__human_peak_model.pdf"
    plt.savefig(out_png1, dpi=FIG_DPI, bbox_inches="tight")
    plt.savefig(out_pdf1, bbox_inches="tight")
    plt.close()
    print("Saved:", out_png1)
    print("Saved:", out_pdf1)

    # plot 2
    plt.figure(figsize=(8, 4))
    sub = summary_df[summary_df["peak_model_used"] == "macaque_peak_model"].copy()
    x = np.arange(sub.shape[0])
    plt.bar(x - 0.15, sub["pred_l2_ref"], width=0.3, label="ref")
    plt.bar(x + 0.15, sub["pred_l2_mut"], width=0.3, label="mut")
    plt.xticks(x, sub["sequence_name"])
    plt.ylabel("Prediction vector L2")
    plt.title(f"{har_id} | {model_label} | macaque peak model")
    plt.legend(frameon=False)
    plt.tight_layout()
    out_png2 = str(prefix) + "__macaque_peak_model.png"
    out_pdf2 = str(prefix) + "__macaque_peak_model.pdf"
    plt.savefig(out_png2, dpi=FIG_DPI, bbox_inches="tight")
    plt.savefig(out_pdf2, bbox_inches="tight")
    plt.close()
    print("Saved:", out_png2)
    print("Saved:", out_pdf2)

    torch.cuda.empty_cache()
    gc.collect()

    return summary_df


# ============================================================
# main
# ============================================================
def main():
    if not PAIRED_CSV.exists():
        raise FileNotFoundError(f"Missing paired csv: {PAIRED_CSV}")
    if not ALIGN_CSV.exists():
        raise FileNotFoundError(f"Missing alignment csv: {ALIGN_CSV}")

    pair_df = pd.read_csv(PAIRED_CSV)
    align_df = pd.read_csv(ALIGN_CSV)

    keep_set = set(KEEP_WINDOWS)
    pair_df = pair_df[
        pair_df.apply(
            lambda r: (str(r["har_id"]), str(r["model_label"]), int(r["window_rank"])) in keep_set,
            axis=1
        )
    ].copy()

    pair_df = add_har1_rows_if_present(pair_df)

    if pair_df.empty:
        raise ValueError("No rows matched KEEP_WINDOWS / HAR1A/HAR1B rules")

    merge_cols = ["har_id", "model_label", "window_rank", "pair_record_id"]
    need_cols = merge_cols + ["human_aln", "macaque_aln"]
    missing = [c for c in need_cols if c not in align_df.columns]
    if missing:
        raise ValueError(f"Alignment csv missing columns: {missing}")

    df = pair_df.merge(
        align_df[need_cols],
        on=merge_cols,
        how="left"
    )

    if df["human_aln"].isna().any() or df["macaque_aln"].isna().any():
        missing_rows = df[df["human_aln"].isna() | df["macaque_aln"].isna()][merge_cols]
        raise ValueError(f"Some selected rows could not find alignment results:\n{missing_rows}")

    print("Selected rows:", df.shape[0])

    human_peak_model, macaque_peak_model = load_peak_models()

    all_summary = []
    for _, row in df.iterrows():
        s = run_single_case(row, human_peak_model, macaque_peak_model)
        all_summary.append(s)

    if len(all_summary) > 0:
        big_summary = pd.concat(all_summary, axis=0, ignore_index=True)
        out_summary = OUT_DIR / "35_peak_model_validation_summary.csv"
        big_summary.to_csv(out_summary, index=False)
        print("\nSaved:", out_summary)

    print("Done.")


if __name__ == "__main__":
    main()