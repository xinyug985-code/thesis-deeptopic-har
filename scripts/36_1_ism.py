#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import os
os.environ["KERAS_BACKEND"] = "torch"

from pathlib import Path
import gc

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import keras
import anndata as ad
import crested
import torch


# ============================================================
# config
# ============================================================
HAR_ID = "HARsv2_1687"

BASE_DIR = Path("/mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/pjt")
CASE_DIR = BASE_DIR / "runs" / "out" / "har1687_case"
OUT_DIR = CASE_DIR / "36_3_human_focus_ism"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FASTA_500 = CASE_DIR / f"{HAR_ID}__500bp.fa"
HUMAN_WINDOWS_CSV = CASE_DIR / f"{HAR_ID}__human__top_windows.csv"

HUMAN_TOPIC_MODEL_PATH = BASE_DIR / "runs" / "out" / "deeptopic_human" / "final_model.keras"
MACAQUE_TOPIC_MODEL_PATH = BASE_DIR / "runs" / "out" / "deeptopic_macaque" / "final_model.keras"

HUMAN_ADATA_PATH = BASE_DIR / "runs" / "out" / "human_topics.h5ad"
MACAQUE_ADATA_PATH = BASE_DIR / "runs" / "out" / "macaque_topics.h5ad"

FIG_DPI = 300
BASES = ["A", "C", "G", "T"]

# 只做这几个窗口
KEEP_WINDOWS = [
    ("human_model", 1),
    ("human_model", 2),
    ("macaque_model", 1),
]


# ============================================================
# helpers
# ============================================================
def get_class_names(adata_path: Path) -> list[str]:
    adata = ad.read_h5ad(adata_path)
    names = list(adata.obs_names)
    if len(names) == 0:
        raise ValueError(f"No class names found in {adata_path}")
    return names


def get_target_idx(class_names: list[str], target_topic: str) -> int:
    if target_topic not in class_names:
        raise ValueError(
            f"Target topic {target_topic} not found. "
            f"First 10 names: {class_names[:10]}"
        )
    return class_names.index(target_topic)


def read_fasta_as_dict(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing fasta: {path}")

    seqs = {}
    current_name = None
    current_seq = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_name is not None:
                    seqs[current_name] = "".join(current_seq).upper()
                current_name = line[1:]
                current_seq = []
            else:
                current_seq.append(line)

    if current_name is not None:
        seqs[current_name] = "".join(current_seq).upper()

    return seqs


def predict_sequence(sequence: str, model) -> np.ndarray:
    pred = crested.tl.predict(sequence, model=model)
    pred = np.asarray(pred)

    if pred.ndim == 2 and pred.shape[0] == 1:
        pred = pred[0]
    elif pred.ndim > 2:
        pred = np.squeeze(pred)

    if pred.ndim != 1:
        raise ValueError(f"Unexpected prediction shape: {pred.shape}")

    return pred


def mutate_base(seq: str, pos: int, new_base: str) -> str:
    s = list(seq)
    s[pos] = new_base
    return "".join(s)


def run_ism_for_window(
    full_seq: str,
    model,
    class_names: list[str],
    target_topic: str,
    har_id: str,
    seq_label: str,
    model_label: str,
    window_rank: int,
    window_start: int,
    window_end: int,
    out_prefix: Path,
):
    target_idx = get_target_idx(class_names, target_topic)

    original_pred = predict_sequence(full_seq, model=model)
    original_score = float(original_pred[target_idx])

    rows = []
    for pos in range(window_start, window_end):
        ref_base = full_seq[pos]
        if ref_base not in BASES:
            continue

        for alt_base in BASES:
            if alt_base == ref_base:
                continue

            mutated_seq = mutate_base(full_seq, pos, alt_base)
            mutated_pred = predict_sequence(mutated_seq, model=model)
            mutated_score = float(mutated_pred[target_idx])

            rows.append({
                "har_id": har_id,
                "seq_label": seq_label,
                "model_label": model_label,
                "window_rank": window_rank,
                "target_topic": target_topic,
                "target_idx": target_idx,

                "window_start_in_seq_0based": window_start,
                "window_end_in_seq_0based_exclusive": window_end,
                "window_length": window_end - window_start,

                "position_in_full_0based": pos,
                "position_in_window_0based": pos - window_start,
                "position_in_window_1based": pos - window_start + 1,
                "ref_base": ref_base,
                "alt_base": alt_base,

                "original_score": original_score,
                "mutated_score": mutated_score,
                "score_delta": mutated_score - original_score,
                "score_drop": original_score - mutated_score,
            })

    mut_df = pd.DataFrame(rows)

    summary_df = (
        mut_df.groupby(["position_in_window_0based", "position_in_window_1based", "ref_base"], as_index=False)
        .agg(
            max_score_drop=("score_drop", "max"),
            mean_score_drop=("score_drop", "mean"),
            min_score_drop=("score_drop", "min"),
        )
        .sort_values("position_in_window_0based")
        .reset_index(drop=True)
    )

    idx = mut_df.groupby("position_in_window_0based")["score_drop"].idxmax()
    best_alt_df = mut_df.loc[idx, [
        "position_in_window_0based", "position_in_window_1based", "ref_base",
        "alt_base", "mutated_score", "score_drop"
    ]].copy()

    best_alt_df = best_alt_df.rename(columns={
        "alt_base": "worst_alt_base",
        "mutated_score": "worst_alt_mutated_score",
        "score_drop": "worst_alt_score_drop",
    })

    summary_df = summary_df.merge(
        best_alt_df,
        on=["position_in_window_0based", "position_in_window_1based", "ref_base"],
        how="left"
    )

    mut_csv = str(out_prefix) + "__mutations.csv"
    sum_csv = str(out_prefix) + "__summary.csv"
    mut_df.to_csv(mut_csv, index=False)
    summary_df.to_csv(sum_csv, index=False)

    print("Saved:", mut_csv)
    print("Saved:", sum_csv)

    plt.figure(figsize=(12, 4))
    plt.plot(summary_df["position_in_window_0based"], summary_df["max_score_drop"], label="max score drop")
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("Position in 30-bp window")
    plt.ylabel("Score drop")
    plt.title(f"{har_id} | {seq_label} | {model_label} | {target_topic}")
    plt.legend(frameon=False)
    plt.tight_layout()

    out_png = str(out_prefix) + "__ism.png"
    out_pdf = str(out_prefix) + "__ism.pdf"
    plt.savefig(out_png, dpi=FIG_DPI, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()

    print("Saved:", out_png)
    print("Saved:", out_pdf)

    torch.cuda.empty_cache()
    gc.collect()

    return {
        "original_score": original_score,
        "max_position_drop": float(summary_df["max_score_drop"].max()),
        "mean_position_drop_mean": float(summary_df["mean_score_drop"].mean()),
    }


# ============================================================
# main
# ============================================================
def main():
    seqs = read_fasta_as_dict(FASTA_500)

    # 取 human 500bp
    human_key = [k for k in seqs if "|human|" in k]
    if len(human_key) == 0:
        raise ValueError("No human sequence found in fasta")
    human_seq = seqs[human_key[0]]

    win_df = pd.read_csv(HUMAN_WINDOWS_CSV)
    win_df = win_df[
        win_df.apply(lambda r: (str(r["model_label"]), int(r["window_rank"])) in set(KEEP_WINDOWS), axis=1)
    ].copy()

    if win_df.empty:
        raise ValueError("No windows left after KEEP_WINDOWS filtering")

    human_topic_model = keras.models.load_model(str(HUMAN_TOPIC_MODEL_PATH), compile=False)
    macaque_topic_model = keras.models.load_model(str(MACAQUE_TOPIC_MODEL_PATH), compile=False)

    human_class_names = get_class_names(HUMAN_ADATA_PATH)
    macaque_class_names = get_class_names(MACAQUE_ADATA_PATH)

    summary_rows = []

    for _, row in win_df.iterrows():
        model_label = str(row["model_label"])
        window_rank = int(row["window_rank"])
        window_start = int(row["start_in_seq_0based"])
        window_end = int(row["end_in_seq_0based_exclusive"])

        out_prefix = OUT_DIR / f"{HAR_ID}__humanSeq__{model_label}__rank{window_rank}"

        # 用对应模型先预测 full seq 以确定 target topic
        if model_label == "human_model":
            model = human_topic_model
            class_names = human_class_names
        else:
            model = macaque_topic_model
            class_names = macaque_class_names

        pred = predict_sequence(human_seq, model=model)
        target_topic = class_names[int(np.argmax(pred))]

        res = run_ism_for_window(
            full_seq=human_seq,
            model=model,
            class_names=class_names,
            target_topic=target_topic,
            har_id=HAR_ID,
            seq_label="human_seq",
            model_label=model_label,
            window_rank=window_rank,
            window_start=window_start,
            window_end=window_end,
            out_prefix=out_prefix,
        )

        summary_rows.append({
            "har_id": HAR_ID,
            "seq_label": "human_seq",
            "model_label": model_label,
            "window_rank": window_rank,
            "window_start_in_seq_0based": window_start,
            "window_end_in_seq_0based_exclusive": window_end,
            "target_topic": target_topic,
            "original_score": res["original_score"],
            "max_position_drop": res["max_position_drop"],
            "mean_position_drop_mean": res["mean_position_drop_mean"],
        })

    out_summary = OUT_DIR / f"{HAR_ID}__human_focus_ism_summary.csv"
    pd.DataFrame(summary_rows).to_csv(out_summary, index=False)
    print("Saved:", out_summary)
    print("Done.")


if __name__ == "__main__":
    main()