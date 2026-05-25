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
# config
# ============================================================
HAR_ID = "HARsv2_1687"
TARGET_LEN = 500
TOP_K = 3
MIN_SEPARATION = 20
FIG_DPI = 300

BASE_DIR = Path("/mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/pjt")
OUT_DIR = BASE_DIR / "runs" / "out" / "har1687_case"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# input beds
HUMAN_BED = BASE_DIR / "data" / "active_HARs_qc_pass.bed"
MACAQUE_BED = BASE_DIR / "data" / "output_rheMac10.bed"

# topic models
HUMAN_TOPIC_MODEL_PATH = BASE_DIR / "runs" / "out" / "deeptopic_human" / "final_model.keras"
MACAQUE_TOPIC_MODEL_PATH = BASE_DIR / "runs" / "out" / "deeptopic_macaque" / "final_model.keras"

HUMAN_ADATA_PATH = BASE_DIR / "runs" / "out" / "human_topics.h5ad"
MACAQUE_ADATA_PATH = BASE_DIR / "runs" / "out" / "macaque_topics.h5ad"

# genomes
HUMAN_FA = Path("/mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/genomes/hg38/hg38.fa")
HUMAN_CHROMSIZES = Path("/mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/genomes/hg38/hg38.chrom.sizes")

MACAQUE_FA = Path("/mimer/NOBACKUP/groups/naiss2025-22-612/yiquan/macaque/rheMac10.fa")
MACAQUE_CHROMSIZES = Path("/mimer/NOBACKUP/groups/naiss2025-22-612/yiquan/macaque/rheMac10.chrom.sizes")


# ============================================================
# helpers
# ============================================================
def sanitize_name(x: str) -> str:
    x = str(x)
    for ch in [" ", "/", "\\", "|", ":", ";", ",", "(", ")", "[", "]"]:
        x = x.replace(ch, "_")
    return x


def get_class_names(adata_path: Path) -> list[str]:
    adata = ad.read_h5ad(adata_path)
    names = list(adata.obs_names)
    if len(names) == 0:
        raise ValueError(f"No class names found in {adata_path}")
    return names


def get_target_idx(class_names: list[str], target_topic: str) -> int:
    if target_topic not in class_names:
        raise ValueError(
            f"Target topic {target_topic} not found in class_names. "
            f"First 10 names: {class_names[:10]}"
        )
    return class_names.index(target_topic)


def resize_to_center_window(chrom: str, start: int, end: int, target_len: int = 500):
    mid = (int(start) + int(end)) // 2
    seq_start = int(mid - target_len // 2)
    seq_end = int(seq_start + target_len)
    if seq_start < 0:
        seq_start = 0
        seq_end = target_len
    return seq_start, seq_end


def fetch_seq(genome: crested.Genome, chrom: str, start: int, end: int) -> str:
    return genome.fetch(chrom, int(start), int(end)).upper()


def load_har_from_bed(har_id: str, bed_path: Path) -> pd.Series:
    if not bed_path.exists():
        raise FileNotFoundError(f"Missing bed: {bed_path}")

    har = pd.read_csv(bed_path, sep="\t", header=None, comment="#")
    if har.shape[1] < 3:
        raise ValueError(f"{bed_path} has fewer than 3 columns")

    if har.shape[1] >= 4:
        har = har.iloc[:, :4].copy()
        har.columns = ["chrom", "start", "end", "name"]
    else:
        har = har.iloc[:, :3].copy()
        har.columns = ["chrom", "start", "end"]
        har["name"] = (
            har["chrom"].astype(str)
            + ":"
            + har["start"].astype(str)
            + "-"
            + har["end"].astype(str)
        )

    hit = har[har["name"].astype(str) == str(har_id)].copy()
    if hit.empty:
        raise ValueError(f"{har_id} not found in {bed_path}")

    if hit.shape[0] > 1:
        warnings.warn(f"Multiple rows found for {har_id}, taking first one")

    return hit.iloc[0]


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


def summarize_prediction(pred: np.ndarray, class_names: list[str]) -> dict:
    top_idx = int(np.argmax(pred))
    top_score = float(pred[top_idx])
    second_score = float(np.sort(pred)[-2]) if pred.size >= 2 else np.nan

    eps = 1e-12
    p = pred / (pred.sum() + eps)
    entropy = float(-(p * np.log2(p + eps)).sum())

    return {
        "top_idx": top_idx,
        "top_topic": class_names[top_idx],
        "top_score": top_score,
        "second_score": second_score,
        "margin_top1_top2": top_score - second_score if pred.size >= 2 else np.nan,
        "entropy": entropy,
    }


def extract_plot_matrix(scores: np.ndarray) -> np.ndarray:
    arr = np.asarray(scores)

    if arr.ndim == 4:
        return arr[0, 0, :, :]
    if arr.ndim == 3:
        if arr.shape[0] == 1:
            return arr[0, :, :]
        return arr.mean(axis=0)

    raise ValueError(f"Unexpected scores shape: {arr.shape}")


def compute_profile(arr_plot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    profile_abs = np.sum(np.abs(arr_plot), axis=1)
    profile_signed = np.sum(arr_plot, axis=1)
    return profile_abs, profile_signed


def greedy_top_nonoverlap_windows(
    values: np.ndarray,
    window_size: int = 30,
    top_k: int = 3,
    min_separation: int = 20,
):
    n = len(values)
    if n < window_size:
        return []

    window_scores = []
    current = values[:window_size].sum()
    window_scores.append(current)

    for i in range(1, n - window_size + 1):
        current = current - values[i - 1] + values[i + window_size - 1]
        window_scores.append(current)

    candidates = [
        {"start": i, "end": i + window_size, "score": float(window_scores[i])}
        for i in range(len(window_scores))
    ]
    candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)

    selected = []
    for cand in candidates:
        ok = True
        for s in selected:
            if not (cand["end"] + min_separation <= s["start"] or cand["start"] >= s["end"] + min_separation):
                ok = False
                break
        if ok:
            selected.append(cand)
        if len(selected) >= top_k:
            break

    for rank, s in enumerate(selected, start=1):
        s["rank"] = rank

    return selected


def run_one_species(
    species_label: str,
    source_bed: str,
    chrom: str,
    har_start: int,
    har_end: int,
    genome: crested.Genome,
    human_topic_model,
    macaque_topic_model,
    human_class_names: list[str],
    macaque_class_names: list[str],
):
    seq_start, seq_end = resize_to_center_window(chrom, har_start, har_end, TARGET_LEN)
    seq = fetch_seq(genome, chrom, seq_start, seq_end)

    if len(seq) != TARGET_LEN:
        raise ValueError(f"{species_label} fetched seq len {len(seq)} != {TARGET_LEN}")

    pred_h = predict_sequence(seq, human_topic_model)
    pred_m = predict_sequence(seq, macaque_topic_model)

    sum_h = summarize_prediction(pred_h, human_class_names)
    sum_m = summarize_prediction(pred_m, macaque_class_names)

    target_idx_h = get_target_idx(human_class_names, sum_h["top_topic"])
    scores_h, onehot_h = crested.tl.contribution_scores(
        seq,
        target_idx=target_idx_h,
        model=human_topic_model,
    )

    target_idx_m = get_target_idx(macaque_class_names, sum_m["top_topic"])
    scores_m, onehot_m = crested.tl.contribution_scores(
        seq,
        target_idx=target_idx_m,
        model=macaque_topic_model,
    )

    arr_plot_h = extract_plot_matrix(scores_h)
    prof_abs_h, prof_signed_h = compute_profile(arr_plot_h)

    arr_plot_m = extract_plot_matrix(scores_m)
    prof_abs_m, prof_signed_m = compute_profile(arr_plot_m)

    windows_h = greedy_top_nonoverlap_windows(
        prof_abs_h, window_size=30, top_k=TOP_K, min_separation=MIN_SEPARATION
    )
    windows_m = greedy_top_nonoverlap_windows(
        prof_abs_m, window_size=30, top_k=TOP_K, min_separation=MIN_SEPARATION
    )

    return {
        "species": species_label,
        "source_bed": source_bed,
        "chrom": chrom,
        "har_start": har_start,
        "har_end": har_end,
        "seq_start": seq_start,
        "seq_end": seq_end,
        "sequence": seq,

        "pred_h": pred_h,
        "pred_m": pred_m,
        "summary_h": sum_h,
        "summary_m": sum_m,

        "scores_h": scores_h,
        "onehot_h": onehot_h,
        "scores_m": scores_m,
        "onehot_m": onehot_m,

        "prof_abs_h": prof_abs_h,
        "prof_signed_h": prof_signed_h,
        "prof_abs_m": prof_abs_m,
        "prof_signed_m": prof_signed_m,

        "windows_h": windows_h,
        "windows_m": windows_m,
    }


def save_prediction_csv(prefix: Path, species_result: dict, human_class_names, macaque_class_names):
    df_h = pd.DataFrame({
        "topic": human_class_names,
        "score": species_result["pred_h"],
    }).sort_values("score", ascending=False).reset_index(drop=True)

    df_m = pd.DataFrame({
        "topic": macaque_class_names,
        "score": species_result["pred_m"],
    }).sort_values("score", ascending=False).reset_index(drop=True)

    out_h = str(prefix) + "__human_model_predictions.csv"
    out_m = str(prefix) + "__macaque_model_predictions.csv"
    df_h.to_csv(out_h, index=False)
    df_m.to_csv(out_m, index=False)
    print("Saved:", out_h)
    print("Saved:", out_m)


def save_barplot(prefix: Path, pred: np.ndarray, class_names: list[str], title: str):
    df = pd.DataFrame({"topic": class_names, "score": pred}).sort_values("score", ascending=False).reset_index(drop=True)
    topn = min(15, df.shape[0])

    plt.figure(figsize=(8, 4))
    plt.bar(np.arange(topn), df["score"].iloc[:topn])
    plt.xticks(np.arange(topn), df["topic"].iloc[:topn], rotation=90)
    plt.ylabel("Score")
    plt.title(title)
    plt.tight_layout()

    out_png = str(prefix) + ".png"
    out_pdf = str(prefix) + ".pdf"
    plt.savefig(out_png, dpi=FIG_DPI, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    print("Saved:", out_png)
    print("Saved:", out_pdf)


def save_contribution_plot(prefix: Path, scores, onehot, seq_label: str, class_label: str, title: str):
    plt.figure(figsize=(14, 4))
    crested.pl.patterns.contribution_scores(
        scores,
        onehot,
        sequence_labels=[seq_label],
        class_labels=[class_label],
        zoom_n_bases=500,
        title=title,
        height=3,
    )
    plt.tight_layout()

    out_png = str(prefix) + ".png"
    out_pdf = str(prefix) + ".pdf"
    plt.savefig(out_png, dpi=FIG_DPI, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    print("Saved:", out_png)
    print("Saved:", out_pdf)


def save_profile_plot(prefix: Path, prof_abs: np.ndarray, prof_signed: np.ndarray, title: str):
    plt.figure(figsize=(12, 3.5))
    plt.plot(prof_abs, label="abs contribution")
    plt.plot(prof_signed, label="signed contribution")
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("Position")
    plt.ylabel("Contribution")
    plt.title(title)
    plt.legend(frameon=False)
    plt.tight_layout()

    out_png = str(prefix) + ".png"
    out_pdf = str(prefix) + ".pdf"
    plt.savefig(out_png, dpi=FIG_DPI, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    print("Saved:", out_png)
    print("Saved:", out_pdf)


def save_windows_csv(prefix: Path, species_result: dict):
    rows = []

    for model_label, windows in [
        ("human_model", species_result["windows_h"]),
        ("macaque_model", species_result["windows_m"]),
    ]:
        for w in windows:
            s = int(w["start"])
            e = int(w["end"])
            seq = species_result["sequence"][s:e]

            rows.append({
                "har_id": HAR_ID,
                "species": species_result["species"],
                "source_bed": species_result["source_bed"],
                "model_label": model_label,
                "window_rank": int(w["rank"]),
                "window_score_sum_abs": float(w["score"]),
                "window_size": e - s,
                "start_in_seq_0based": s,
                "end_in_seq_0based_exclusive": e,
                "chrom": species_result["chrom"],
                "genomic_start_0based": species_result["seq_start"] + s,
                "genomic_end_0based_exclusive": species_result["seq_start"] + e,
                "sequence": seq,
            })

    out_df = pd.DataFrame(rows).sort_values(["species", "model_label", "window_rank"]).reset_index(drop=True)
    out_csv = str(prefix) + "__top_windows.csv"
    out_df.to_csv(out_csv, index=False)
    print("Saved:", out_csv)
    return out_df


# ============================================================
# main
# ============================================================
def main():
    print(f"========== single-case mini pipeline for {HAR_ID} ==========")

    human_topic_model = keras.models.load_model(str(HUMAN_TOPIC_MODEL_PATH), compile=False)
    macaque_topic_model = keras.models.load_model(str(MACAQUE_TOPIC_MODEL_PATH), compile=False)

    human_class_names = get_class_names(HUMAN_ADATA_PATH)
    macaque_class_names = get_class_names(MACAQUE_ADATA_PATH)

    human_genome = crested.Genome(HUMAN_FA, HUMAN_CHROMSIZES)
    macaque_genome = crested.Genome(MACAQUE_FA, MACAQUE_CHROMSIZES)

    human_row = load_har_from_bed(HAR_ID, HUMAN_BED)
    macaque_row = load_har_from_bed(HAR_ID, MACAQUE_BED)

    human_chrom = str(human_row["chrom"])
    human_start = int(human_row["start"])
    human_end = int(human_row["end"])

    macaque_chrom = str(macaque_row["chrom"])
    macaque_start = int(macaque_row["start"])
    macaque_end = int(macaque_row["end"])

    human_res = run_one_species(
        species_label="human",
        source_bed=str(HUMAN_BED),
        chrom=human_chrom,
        har_start=human_start,
        har_end=human_end,
        genome=human_genome,
        human_topic_model=human_topic_model,
        macaque_topic_model=macaque_topic_model,
        human_class_names=human_class_names,
        macaque_class_names=macaque_class_names,
    )

    macaque_res = run_one_species(
        species_label="macaque",
        source_bed=str(MACAQUE_BED),
        chrom=macaque_chrom,
        har_start=macaque_start,
        har_end=macaque_end,
        genome=macaque_genome,
        human_topic_model=human_topic_model,
        macaque_topic_model=macaque_topic_model,
        human_class_names=human_class_names,
        macaque_class_names=macaque_class_names,
    )

    summary_rows = []
    for res in [human_res, macaque_res]:
        summary_rows.append({
            "har_id": HAR_ID,
            "species": res["species"],
            "source_bed": res["source_bed"],
            "chrom": res["chrom"],
            "har_start": res["har_start"],
            "har_end": res["har_end"],
            "seq_start": res["seq_start"],
            "seq_end": res["seq_end"],

            "human_model_top_topic": res["summary_h"]["top_topic"],
            "human_model_top_score": res["summary_h"]["top_score"],
            "human_model_second_score": res["summary_h"]["second_score"],
            "human_model_margin_top1_top2": res["summary_h"]["margin_top1_top2"],
            "human_model_entropy": res["summary_h"]["entropy"],

            "macaque_model_top_topic": res["summary_m"]["top_topic"],
            "macaque_model_top_score": res["summary_m"]["top_score"],
            "macaque_model_second_score": res["summary_m"]["second_score"],
            "macaque_model_margin_top1_top2": res["summary_m"]["margin_top1_top2"],
            "macaque_model_entropy": res["summary_m"]["entropy"],
        })

    summary_df = pd.DataFrame(summary_rows)
    out_summary = OUT_DIR / f"{HAR_ID}__summary.csv"
    summary_df.to_csv(out_summary, index=False)
    print("Saved:", out_summary)

    out_fasta = OUT_DIR / f"{HAR_ID}__500bp.fa"
    with open(out_fasta, "w") as f:
        f.write(f">{HAR_ID}|human|{human_res['chrom']}:{human_res['seq_start']}-{human_res['seq_end']}\n")
        f.write(human_res["sequence"] + "\n")
        f.write(f">{HAR_ID}|macaque|{macaque_res['chrom']}:{macaque_res['seq_start']}-{macaque_res['seq_end']}\n")
        f.write(macaque_res["sequence"] + "\n")
    print("Saved:", out_fasta)

    for res in [human_res, macaque_res]:
        species = res["species"]
        prefix = OUT_DIR / f"{HAR_ID}__{species}"

        save_prediction_csv(prefix, res, human_class_names, macaque_class_names)

        save_barplot(
            prefix.with_name(prefix.name + "__human_model_bar"),
            res["pred_h"],
            human_class_names,
            f"{HAR_ID} | {species} seq | human model"
        )
        save_barplot(
            prefix.with_name(prefix.name + "__macaque_model_bar"),
            res["pred_m"],
            macaque_class_names,
            f"{HAR_ID} | {species} seq | macaque model"
        )

        save_contribution_plot(
            prefix.with_name(prefix.name + "__human_model_contrib"),
            res["scores_h"],
            res["onehot_h"],
            f"{HAR_ID}|{species}",
            res["summary_h"]["top_topic"],
            f"{HAR_ID} | {species} seq | human model | {res['summary_h']['top_topic']}"
        )
        save_contribution_plot(
            prefix.with_name(prefix.name + "__macaque_model_contrib"),
            res["scores_m"],
            res["onehot_m"],
            f"{HAR_ID}|{species}",
            res["summary_m"]["top_topic"],
            f"{HAR_ID} | {species} seq | macaque model | {res['summary_m']['top_topic']}"
        )

        save_profile_plot(
            prefix.with_name(prefix.name + "__human_model_profile"),
            res["prof_abs_h"],
            res["prof_signed_h"],
            f"{HAR_ID} | {species} seq | human model profile"
        )
        save_profile_plot(
            prefix.with_name(prefix.name + "__macaque_model_profile"),
            res["prof_abs_m"],
            res["prof_signed_m"],
            f"{HAR_ID} | {species} seq | macaque model profile"
        )

        save_windows_csv(prefix, res)

    print("\nDone.")
    print("Outputs:", OUT_DIR)


if __name__ == "__main__":
    main()