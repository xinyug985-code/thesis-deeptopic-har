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
FIG_DPI = 300
TOP_N_MOTIFS = 5
LABEL_TOP_N = 3

BASE_DIR = Path("/mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/pjt")
CASE_DIR = BASE_DIR / "runs" / "out" / "har1687_case"
OUT_DIR = CASE_DIR / "36_4_highlight_plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_CSV = CASE_DIR / f"{HAR_ID}__summary.csv"
FASTA_500 = CASE_DIR / f"{HAR_ID}__500bp.fa"

HUMAN_WINDOWS_CSV = CASE_DIR / f"{HAR_ID}__human__top_windows.csv"
MACAQUE_WINDOWS_CSV = CASE_DIR / f"{HAR_ID}__macaque__top_windows.csv"

HUMAN_TOPIC_MODEL_PATH = BASE_DIR / "runs" / "out" / "deeptopic_human" / "final_model.keras"
MACAQUE_TOPIC_MODEL_PATH = BASE_DIR / "runs" / "out" / "deeptopic_macaque" / "final_model.keras"

HUMAN_ADATA_PATH = BASE_DIR / "runs" / "out" / "human_topics.h5ad"
MACAQUE_ADATA_PATH = BASE_DIR / "runs" / "out" / "macaque_topics.h5ad"

FIMO_CSV = CASE_DIR / "HARsv2_1687__fimo_layout_top5_allMotifs.csv"


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


def choose_rank_col(df: pd.DataFrame) -> str | None:
    candidates = [
        "window_score_sum_abs",
        "layout_score",
        "window_max_abs",
        "window_mean_abs",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    if "p-value" in df.columns:
        return "p-value"
    return None


def get_top_motif_hits(fimo_df: pd.DataFrame, species: str, seq_start: int, seq_end: int, top_n: int = 5) -> pd.DataFrame:
    # 兼容旧列名 species 和新列名 seq_species
    species_col = None
    if "species" in fimo_df.columns:
        species_col = "species"
    elif "seq_species" in fimo_df.columns:
        species_col = "seq_species"
    else:
        raise ValueError(f"FIMO table missing species column. Columns are: {fimo_df.columns.tolist()}")

    sub = fimo_df[fimo_df[species_col].astype(str) == str(species)].copy()
    if sub.empty:
        return sub

    sub = sub[(pd.to_numeric(sub["start"], errors="coerce") >= seq_start) &
              (pd.to_numeric(sub["stop"], errors="coerce") <= seq_end)].copy()
    if sub.empty:
        return sub

    rank_col = choose_rank_col(sub)
    if rank_col is None:
        return sub.head(top_n).copy()

    ascending = True if rank_col == "p-value" else False
    sub = sub.sort_values(rank_col, ascending=ascending).copy()

    if "tf_name" in sub.columns:
        sub = sub.drop_duplicates(subset=["tf_name"], keep="first").copy()

    return sub.head(top_n).copy()


def draw_har_and_motif_spans(ax, seq_start: int, har_start: int, har_end: int, motif_hits: pd.DataFrame):
    # HAR 灰色
    har_s = har_start - seq_start
    har_e = har_end - seq_start
    if har_e > har_s:
        ax.axvspan(har_s, har_e, color="lightgray", alpha=0.35, lw=0, zorder=0)

    # motif 金色
    if motif_hits is not None and not motif_hits.empty:
        for _, r in motif_hits.iterrows():
            s = int(r["start"]) - seq_start
            e = int(r["stop"]) - seq_start + 1
            if e > s:
                ax.axvspan(s, e, color="gold", alpha=0.22, lw=0, zorder=0)


def add_manual_labels(ax, seq_start: int, seq_end: int, har_start: int, har_end: int, motif_hits: pd.DataFrame):
    seq_len = seq_end - seq_start

    har_s = har_start - seq_start
    har_e = har_end - seq_start
    har_mid = (har_s + har_e) / 2

    ax.text(
        har_mid / seq_len,
        1.08,
        "HAR body",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
        color="dimgray",
        clip_on=False,
    )

    if motif_hits is None or motif_hits.empty:
        return

    y_levels = [1.16, 1.26, 1.36]
    label_df = motif_hits.head(LABEL_TOP_N).copy()

    for i, (_, r) in enumerate(label_df.iterrows()):
        s = int(r["start"]) - seq_start
        e = int(r["stop"]) - seq_start + 1
        mid = (s + e) / 2
        tf = str(r["tf_name"]) if "tf_name" in r.index else "motif"
        y = y_levels[i % len(y_levels)]

        ax.text(
            mid / seq_len,
            y,
            tf,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=25,
            color="black",
            clip_on=False,
        )


# ============================================================
# main
# ============================================================
def main():
    summary_df = pd.read_csv(SUMMARY_CSV)
    seqs = read_fasta_as_dict(FASTA_500)

    human_key = [k for k in seqs if "|human|" in k][0]
    macaque_key = [k for k in seqs if "|macaque|" in k][0]
    human_seq = seqs[human_key]
    macaque_seq = seqs[macaque_key]

    human_row = summary_df[summary_df["species"] == "human"].iloc[0]
    macaque_row = summary_df[summary_df["species"] == "macaque"].iloc[0]

    human_topic_model = keras.models.load_model(str(HUMAN_TOPIC_MODEL_PATH), compile=False)
    macaque_topic_model = keras.models.load_model(str(MACAQUE_TOPIC_MODEL_PATH), compile=False)

    human_class_names = get_class_names(HUMAN_ADATA_PATH)
    macaque_class_names = get_class_names(MACAQUE_ADATA_PATH)

    # FIMO optional
    fimo_df = pd.read_csv(FIMO_CSV) if FIMO_CSV.exists() else pd.DataFrame()

    plot_jobs = [
        {
            "species": "human",
            "seq": human_seq,
            "row": human_row,
            "model_label": "human_model",
            "model": human_topic_model,
            "class_names": human_class_names,
            "target_topic": str(human_row["human_model_top_topic"]),
        },
        {
            "species": "human",
            "seq": human_seq,
            "row": human_row,
            "model_label": "macaque_model",
            "model": macaque_topic_model,
            "class_names": macaque_class_names,
            "target_topic": str(human_row["macaque_model_top_topic"]),
        },
        {
            "species": "macaque",
            "seq": macaque_seq,
            "row": macaque_row,
            "model_label": "human_model",
            "model": human_topic_model,
            "class_names": human_class_names,
            "target_topic": str(macaque_row["human_model_top_topic"]),
        },
        {
            "species": "macaque",
            "seq": macaque_seq,
            "row": macaque_row,
            "model_label": "macaque_model",
            "model": macaque_topic_model,
            "class_names": macaque_class_names,
            "target_topic": str(macaque_row["macaque_model_top_topic"]),
        },
    ]

    for job in plot_jobs:
        species = job["species"]
        seq = job["seq"]
        row = job["row"]
        model_label = job["model_label"]
        model = job["model"]
        class_names = job["class_names"]
        target_topic = job["target_topic"]

        seq_start = int(row["seq_start"])
        seq_end = int(row["seq_end"])
        har_start = int(row["har_start"])
        har_end = int(row["har_end"])

        target_idx = get_target_idx(class_names, target_topic)
        scores, onehot = crested.tl.contribution_scores(
            seq,
            target_idx=target_idx,
            model=model,
        )

        motif_hits = pd.DataFrame()
        if not fimo_df.empty:
            motif_hits = get_top_motif_hits(
                fimo_df=fimo_df,
                species=species,
                seq_start=seq_start,
                seq_end=seq_end,
                top_n=TOP_N_MOTIFS,
            )

        plt.figure(figsize=(16, 4.8))
        crested.pl.patterns.contribution_scores(
            scores,
            onehot,
            sequence_labels=[f"{HAR_ID}|{species}"],
            class_labels=[target_topic],
            zoom_n_bases=500,
            title=f"{HAR_ID} | {species} seq | {model_label} | {target_topic}",
            height=3,
        )

        ax = plt.gca()
        draw_har_and_motif_spans(ax, seq_start, har_start, har_end, motif_hits)
        add_manual_labels(ax, seq_start, seq_end, har_start, har_end, motif_hits)

        plt.tight_layout()
        out_png = OUT_DIR / f"{HAR_ID}__{species}__{model_label}__highlight.png"
        out_pdf = OUT_DIR / f"{HAR_ID}__{species}__{model_label}__highlight.pdf"
        plt.savefig(out_png, dpi=FIG_DPI, bbox_inches="tight")
        plt.savefig(out_pdf, bbox_inches="tight")
        plt.close()

        print("Saved:", out_png)
        print("Saved:", out_pdf)

        if not motif_hits.empty:
            out_hits = OUT_DIR / f"{HAR_ID}__{species}__{model_label}__top_motifs.csv"
            motif_hits.to_csv(out_hits, index=False)
            print("Saved:", out_hits)

        torch.cuda.empty_cache()
        gc.collect()

    print("Done.")


if __name__ == "__main__":
    main()