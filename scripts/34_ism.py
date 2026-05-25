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
import anndata as ad
import crested
import torch


# ============================================================
# paths
# ============================================================
BASE_DIR = Path("/mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/pjt")
CASE_DIR = BASE_DIR / "runs" / "out" / "case6_contribution_crossseq"

PAIRED_CSV = CASE_DIR / "selected5_human_macaque_window_pairs.csv"

HUMAN_MODEL_PATH = BASE_DIR / "runs" / "out" / "deeptopic_human" / "final_model.keras"
MACAQUE_MODEL_PATH = BASE_DIR / "runs" / "out" / "deeptopic_macaque" / "final_model.keras"

HUMAN_ADATA_PATH = BASE_DIR / "runs" / "out" / "human_topics.h5ad"
MACAQUE_ADATA_PATH = BASE_DIR / "runs" / "out" / "macaque_topics.h5ad"

HUMAN_FA = Path("/mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/genomes/hg38/hg38.fa")
MACAQUE_FA = Path("/mimer/NOBACKUP/groups/naiss2025-22-612/yiquan/macaque/rheMac10.fa")

OUT_DIR = CASE_DIR / "34_ism"
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

BASES = ["A", "C", "G", "T"]
FIG_DPI = 300


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


def load_models():
    print("Loading models...")
    human_model = keras.models.load_model(str(HUMAN_MODEL_PATH), compile=False)
    macaque_model = keras.models.load_model(str(MACAQUE_MODEL_PATH), compile=False)
    print("Human model loaded:", HUMAN_MODEL_PATH)
    print("Macaque model loaded:", MACAQUE_MODEL_PATH)
    return human_model, macaque_model


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
            f"First 10 class names: {class_names[:10]}"
        )
    return class_names.index(target_topic)


def get_model_and_classes(model_label: str, human_model, macaque_model, human_class_names, macaque_class_names):
    if model_label == "human_model":
        return human_model, human_class_names
    elif model_label == "macaque_model":
        return macaque_model, macaque_class_names
    else:
        raise ValueError(f"Unsupported model_label: {model_label}")


def get_genome(species_label: str):
    species_label = str(species_label).lower()
    if species_label == "human":
        return crested.Genome(HUMAN_FA)
    elif species_label == "macaque":
        return crested.Genome(MACAQUE_FA)
    else:
        raise ValueError(f"Unsupported species_label: {species_label}")


def fetch_full_sequence_from_region(species_label: str, region: str) -> str:
    genome = get_genome(species_label)
    chrom, start, end = parse_region(region)
    seq = genome.fetch(chrom, int(start), int(end)).upper()
    expected_len = end - start
    if len(seq) != expected_len:
        warnings.warn(
            f"Fetched sequence length {len(seq)} != expected {expected_len} "
            f"for {species_label} {region}"
        )
    return seq


def mutate_base(seq: str, pos: int, new_base: str) -> str:
    s = list(seq)
    s[pos] = new_base
    return "".join(s)


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


def validate_window_matches_full(
    full_seq: str,
    window_seq: str,
    start0: int,
    end0: int,
    species_label: str,
    pair_id: str,
):
    sub = full_seq[start0:end0].upper()
    if sub != str(window_seq).upper():
        raise ValueError(
            f"Window sequence does not match full sequence for {species_label} | {pair_id}\n"
            f"Expected from full seq: {sub}\n"
            f"Window seq in table   : {window_seq}"
        )


def run_ism_for_species(
    row: pd.Series,
    model,
    class_names: list[str],
    species_label: str,
    out_prefix: Path,
):
    """
    species_label: 'human' or 'macaque'
    """
    har_id = str(row["har_id"])
    model_label = str(row["model_label"])
    rank = int(row["window_rank"])
    pair_id = str(row["pair_record_id"])

    region_col = f"region__{species_label}"
    topic_col = f"target_topic__{species_label}"
    seq_col = f"sequence__{species_label}"
    win_s_col = f"start_in_seq_0based__{species_label}"
    win_e_col = f"end_in_seq_0based_exclusive__{species_label}"

    region = str(row[region_col])
    target_topic = str(row[topic_col])
    window_seq = str(row[seq_col]).upper()
    win_s = int(row[win_s_col])
    win_e = int(row[win_e_col])

    full_seq = fetch_full_sequence_from_region(species_label, region).upper()
    validate_window_matches_full(
        full_seq=full_seq,
        window_seq=window_seq,
        start0=win_s,
        end0=win_e,
        species_label=species_label,
        pair_id=pair_id,
    )

    target_idx = get_target_idx(class_names, target_topic)

    print(
        f"[ISM] {har_id} | {species_label} seq | {model_label} | "
        f"rank={rank} | target_topic={target_topic} | target_idx={target_idx} | "
        f"window={win_s}-{win_e} | full_len={len(full_seq)}"
    )

    if len(full_seq) != 500:
        warnings.warn(
            f"Full sequence length is {len(full_seq)}, not 500, for {species_label} {pair_id}"
        )

    original_pred = predict_sequence(full_seq, model=model)
    original_score = float(original_pred[target_idx])

    mut_rows = []

    for pos in range(win_s, win_e):
        ref_base = full_seq[pos]

        if ref_base not in BASES:
            continue

        for alt_base in BASES:
            if alt_base == ref_base:
                continue

            mutated_full_seq = mutate_base(full_seq, pos, alt_base)
            mutated_pred = predict_sequence(mutated_full_seq, model=model)
            mutated_score = float(mutated_pred[target_idx])

            score_delta = mutated_score - original_score
            score_drop = original_score - mutated_score

            mut_rows.append({
                "har_id": har_id,
                "model_label": model_label,
                "window_rank": rank,
                "pair_record_id": pair_id,
                "species_label": species_label,
                "target_topic": target_topic,
                "target_idx": target_idx,

                "region": region,
                "full_sequence_length": len(full_seq),
                "window_start_in_seq_0based": win_s,
                "window_end_in_seq_0based_exclusive": win_e,
                "window_length": win_e - win_s,

                "ref_full_sequence": full_seq,
                "ref_window_sequence": full_seq[win_s:win_e],

                "position_in_full_0based": pos,
                "position_in_window_0based": pos - win_s,
                "position_in_window_1based": pos - win_s + 1,
                "ref_base": ref_base,
                "alt_base": alt_base,

                "original_score": original_score,
                "mutated_score": mutated_score,
                "score_delta": score_delta,
                "score_drop": score_drop,
            })

    mut_df = pd.DataFrame(mut_rows)
    if mut_df.empty:
        raise ValueError(f"No mutation rows generated for {pair_id} | {species_label}")

    summary_df = (
        mut_df.groupby(
            ["position_in_window_0based", "position_in_window_1based", "ref_base"],
            as_index=False
        )
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
        "position_in_window_0based",
        "position_in_window_1based",
        "ref_base",
        "alt_base",
        "mutated_score",
        "score_drop",
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

    mut_csv = str(out_prefix) + f"__{species_label}_seq__mutations.csv"
    sum_csv = str(out_prefix) + f"__{species_label}_seq__summary.csv"
    mut_df.to_csv(mut_csv, index=False)
    summary_df.to_csv(sum_csv, index=False)

    print("Saved:", mut_csv)
    print("Saved:", sum_csv)

    plt.figure(figsize=(12, 4))
    plt.plot(summary_df["position_in_window_0based"], summary_df["max_score_drop"], label="max score drop")
    plt.plot(summary_df["position_in_window_0based"], summary_df["mean_score_drop"], label="mean score drop")
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("Position in 30-bp window")
    plt.ylabel("Score drop")
    plt.title(f"{har_id} | {species_label} seq | {model_label} | {target_topic}")
    plt.legend(frameon=False)
    plt.tight_layout()

    plot_png = str(out_prefix) + f"__{species_label}_seq__ism.png"
    plot_pdf = str(out_prefix) + f"__{species_label}_seq__ism.pdf"
    plt.savefig(plot_png, dpi=FIG_DPI, bbox_inches="tight")
    plt.savefig(plot_pdf, bbox_inches="tight")
    plt.close()

    print("Saved:", plot_png)
    print("Saved:", plot_pdf)

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
    if not PAIRED_CSV.exists():
        raise FileNotFoundError(f"Missing paired csv: {PAIRED_CSV}")

    df = pd.read_csv(PAIRED_CSV)
    print("Loaded paired table:", PAIRED_CSV)
    print("Shape:", df.shape)

    keep_set = set(KEEP_WINDOWS)
    df = df[
        df.apply(
            lambda r: (str(r["har_id"]), str(r["model_label"]), int(r["window_rank"])) in keep_set,
            axis=1
        )
    ].copy()

    if df.empty:
        raise ValueError("No rows matched KEEP_WINDOWS")

    print("Filtered rows:", df.shape[0])

    human_model, macaque_model = load_models()
    human_class_names = get_class_names(HUMAN_ADATA_PATH)
    macaque_class_names = get_class_names(MACAQUE_ADATA_PATH)

    summary_rows = []

    for _, row in df.iterrows():
        har_id = str(row["har_id"])
        model_label = str(row["model_label"])
        rank = int(row["window_rank"])

        out_prefix = OUT_DIR / f"{sanitize_name(har_id)}__{model_label}__rank{rank}"

        print("\n" + "=" * 90)
        print(f"Processing {har_id} | {model_label} | rank{rank}")

        model, class_names = get_model_and_classes(
            model_label,
            human_model,
            macaque_model,
            human_class_names,
            macaque_class_names,
        )

        res_h = run_ism_for_species(
            row=row,
            model=model,
            class_names=class_names,
            species_label="human",
            out_prefix=out_prefix,
        )
        
        res_m = run_ism_for_species(
            row=row,
            model=model,
            class_names=class_names,
            species_label="macaque",
            out_prefix=out_prefix,
        )

        summary_rows.append({
            "har_id": har_id,
            "model_label": model_label,
            "window_rank": rank,
            "pair_record_id": row["pair_record_id"],

            "target_topic__human": row["target_topic__human"],
            "target_topic__macaque": row["target_topic__macaque"],

            "sequence__human": row["sequence__human"],
            "sequence__macaque": row["sequence__macaque"],

            "original_score__human": res_h["original_score"],
            "max_position_drop__human": res_h["max_position_drop"],
            "mean_position_drop_mean__human": res_h["mean_position_drop_mean"],

            "original_score__macaque": res_m["original_score"],
            "max_position_drop__macaque": res_m["max_position_drop"],
            "mean_position_drop_mean__macaque": res_m["mean_position_drop_mean"],
        })

    summary_df = pd.DataFrame(summary_rows)
    out_summary = OUT_DIR / "34_ism_summary.csv"
    summary_df.to_csv(out_summary, index=False)

    print("\nSaved:", out_summary)
    print("Done.")
    print("Main output:", out_summary)


if __name__ == "__main__":
    main()