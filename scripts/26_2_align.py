#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import os
os.environ["KERAS_BACKEND"] = "torch"

from pathlib import Path
import warnings
import gc

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import anndata as ad
import keras
import crested
import torch


# ============================================================
# paths
# ============================================================
BASE_DIR = Path("/mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/pjt")
OUT_BASE = BASE_DIR / "runs" / "out" / "har_integrated"
CASE_DIR = BASE_DIR / "runs" / "out" / "case6_contribution_crossseq"

LONG_CSV = OUT_BASE / "master_har_3species_2models_long.csv"
FIMO_LAYOUT_CSV = CASE_DIR / "selected6_fimo_layout_table.csv"

HUMAN_MODEL_PATH = BASE_DIR / "runs" / "out" / "deeptopic_human" / "final_model.keras"
MACAQUE_MODEL_PATH = BASE_DIR / "runs" / "out" / "deeptopic_macaque" / "final_model.keras"

HUMAN_ADATA_PATH = BASE_DIR / "runs" / "out" / "human_topics.h5ad"
MACAQUE_ADATA_PATH = BASE_DIR / "runs" / "out" / "macaque_topics.h5ad"

HUMAN_FA = Path("/mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/genomes/hg38/hg38.fa")
MACAQUE_FA = Path("/mimer/NOBACKUP/groups/naiss2025-22-612/yiquan/macaque/rheMac10.fa")

OUT_DIR = CASE_DIR / "26_2_har_motif_manual_highlight"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# settings
# ============================================================
SELECTED_CASES = [
    {"har_id": "HARsv2_2699", "group": "human_high"},
    {"har_id": "HARsv2_1082", "group": "human_high"},
    {"har_id": "HARsv2_0513", "group": "human_high"},
    {"har_id": "HARsv2_0920", "group": "macaque_high"},
    {"har_id": "HARsv2_1547", "group": "macaque_high"},
    {"har_id": "HARsv2_0558", "group": "conserved"},
]

PLOT_CONFIGS = [
    {"seq_species": "human", "model_label": "human_model"},
    {"seq_species": "macaque", "model_label": "macaque_model"},
]

ZOOM_N_BASES = 500
FIG_DPI = 300
TOP_N_MOTIFS = 5
LABEL_TOP_N = 3


# ============================================================
# helpers
# ============================================================
def sanitize_name(x: str) -> str:
    x = str(x)
    for ch in [" ", "/", "\\", "|", ":", ";", ",", "(", ")", "[", "]"]:
        x = x.replace(ch, "_")
    return x


def load_long_table() -> pd.DataFrame:
    if not LONG_CSV.exists():
        raise FileNotFoundError(f"Missing file: {LONG_CSV}")
    df = pd.read_csv(LONG_CSV)
    print("Loaded long table:", LONG_CSV)
    print("Shape:", df.shape)
    return df


def load_fimo_layout() -> pd.DataFrame:
    if not FIMO_LAYOUT_CSV.exists():
        raise FileNotFoundError(f"Missing file: {FIMO_LAYOUT_CSV}")
    df = pd.read_csv(FIMO_LAYOUT_CSV)
    print("Loaded FIMO layout:", FIMO_LAYOUT_CSV)
    print("Shape:", df.shape)
    return df


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
            f"Target topic {target_topic} not found in class_names. "
            f"First 10 names: {class_names[:10]}"
        )
    return class_names.index(target_topic)


def get_case_row(long_df: pd.DataFrame, har_id: str, seq_species: str, model_label: str):
    sub = long_df.loc[
        (long_df["har_id"].astype(str) == str(har_id)) &
        (long_df["species"].astype(str) == str(seq_species)) &
        (long_df["model"].astype(str) == str(model_label))
    ].copy()

    if sub.empty:
        return None

    return sub.iloc[0]


def fetch_sequence(row: pd.Series) -> tuple[str, str]:
    chrom = str(row["chrom"])
    seq_start = int(row["seq_start"])
    seq_end = int(row["seq_end"])
    region_str = f"{chrom}:{seq_start}-{seq_end}"

    species = str(row["species"]).lower()
    if species == "human":
        genome = crested.Genome(HUMAN_FA)
    elif species == "macaque":
        genome = crested.Genome(MACAQUE_FA)
    else:
        raise ValueError(f"Unsupported species for fetch: {species}")

    seq = genome.fetch(chrom, seq_start, seq_end).upper()

    if len(seq) != (seq_end - seq_start):
        warnings.warn(
            f"Fetched sequence length {len(seq)} != expected {seq_end - seq_start} "
            f"for {region_str}"
        )
    return region_str, seq


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


def normalize_tf_name(x) -> str:
    if pd.isna(x):
        return "motif"
    return str(x)


def get_top_motif_hits(
    fimo_df: pd.DataFrame,
    har_id: str,
    seq_species: str,
    seq_start: int,
    seq_end: int,
    top_n: int = 5,
) -> pd.DataFrame:
    sub = fimo_df.loc[
        (fimo_df["har_id"].astype(str) == str(har_id)) &
        (fimo_df["seq_species"].astype(str) == str(seq_species))
    ].copy()

    if sub.empty:
        return sub

    if "start" in sub.columns and "stop" in sub.columns:
        sub = sub.loc[
            (sub["start"].astype(int) >= seq_start) &
            (sub["stop"].astype(int) <= seq_end)
        ].copy()

    if sub.empty:
        return sub

    if "tf_name" not in sub.columns:
        sub["tf_name"] = "motif"

    sub["tf_name"] = sub["tf_name"].apply(normalize_tf_name)

    rank_col = choose_rank_col(sub)
    if rank_col is None:
        return sub.head(top_n).copy()

    ascending = True if rank_col == "p-value" else False
    sub = sub.sort_values(rank_col, ascending=ascending).copy()

    sub = sub.drop_duplicates(subset=["tf_name"], keep="first").copy()

    return sub.head(top_n).copy()


def save_top5_table(
    motif_hits: pd.DataFrame,
    out_prefix: Path,
    seq_species: str,
    model_label: str,
):
    if motif_hits is None or motif_hits.empty:
        return

    keep_cols = [c for c in [
        "har_id", "seq_species", "tf_name", "motif_id", "motif_alt_id",
        "start", "stop", "strand", "score", "p-value", "q-value",
        "window_rank", "window_score_sum_abs", "window_max_abs",
        "window_mean_abs", "layout_score"
    ] if c in motif_hits.columns]

    out_csv = str(out_prefix) + f"__{seq_species}_seq__{model_label}__top5_motifs.csv"
    motif_hits.loc[:, keep_cols].to_csv(out_csv, index=False)
    print("Saved:", out_csv)


def draw_har_and_motif_spans(ax, row: pd.Series, motif_hits: pd.DataFrame):
    seq_start = int(row["seq_start"])

    # HAR 灰色
    har_s = int(row["start"]) - seq_start
    har_e = int(row["end"]) - seq_start
    if har_e > har_s:
        ax.axvspan(har_s, har_e, color="lightgray", alpha=0.35, lw=0, zorder=0)

    if motif_hits is not None and not motif_hits.empty:
        for _, r in motif_hits.iterrows():
            try:
                s = int(r["start"]) - seq_start
                e = int(r["stop"]) - seq_start + 1
            except Exception:
                continue

            if e > s:
                ax.axvspan(s, e, color="gold", alpha=0.22, lw=0, zorder=0)


def add_manual_labels(ax, row: pd.Series, motif_hits: pd.DataFrame, label_top_n: int = 3):
    seq_start = int(row["seq_start"])
    seq_end = int(row["seq_end"])
    seq_len = seq_end - seq_start

    # HAR label
    har_s = int(row["start"]) - seq_start
    har_e = int(row["end"]) - seq_start
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

    label_df = motif_hits.head(label_top_n).copy()
    y_levels = [1.16, 1.26, 1.36]

    for i, (_, r) in enumerate(label_df.iterrows()):
        try:
            s = int(r["start"]) - seq_start
            e = int(r["stop"]) - seq_start + 1
        except Exception:
            continue

        tf = normalize_tf_name(r.get("tf_name", "motif"))
        mid = (s + e) / 2
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


def run_contribution_plot_with_manual_highlights(
    sequence: str,
    row: pd.Series,
    motif_hits: pd.DataFrame,
    har_id: str,
    group: str,
    seq_species: str,
    region_str: str,
    model,
    model_label: str,
    target_topic: str,
    class_names: list[str],
    out_prefix: Path,
):
    target_idx = get_target_idx(class_names, target_topic)
    print(
        f"[{har_id}] seq_species={seq_species} | model={model_label} | "
        f"target_topic={target_topic} | target_idx={target_idx}"
    )

    scores, onehot = crested.tl.contribution_scores(
        sequence,
        target_idx=target_idx,
        model=model,
    )

    plt.figure(figsize=(16, 4.8))
    crested.pl.patterns.contribution_scores(
        scores,
        onehot,
        sequence_labels=[f"{har_id} | {seq_species}"],
        class_labels=[target_topic],
        zoom_n_bases=ZOOM_N_BASES,
        title=f"{har_id} | {group} | {seq_species} seq | {model_label} | {target_topic}",
        height=3,
    )

    ax = plt.gca()

    draw_har_and_motif_spans(ax, row, motif_hits)
    add_manual_labels(ax, row, motif_hits, label_top_n=LABEL_TOP_N)

    plt.tight_layout()
    out_png = str(out_prefix) + f"__{seq_species}_seq__{model_label}__har_top5motif_manual.png"
    out_pdf = str(out_prefix) + f"__{seq_species}_seq__{model_label}__har_top5motif_manual.pdf"
    plt.savefig(out_png, dpi=FIG_DPI, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()

    print("Saved:", out_png)
    print("Saved:", out_pdf)

    torch.cuda.empty_cache()
    gc.collect()

    return {
        "target_topic": target_topic,
        "target_idx": target_idx,
        "n_motif_hits": 0 if motif_hits is None else int(motif_hits.shape[0]),
    }


# ============================================================
# main
# ============================================================
def main():
    print("========== 26_2 HAR + top5 motif manual-highlight plots ==========")

    long_df = load_long_table()
    fimo_df = load_fimo_layout()
    human_model, macaque_model = load_models()

    human_class_names = get_class_names(HUMAN_ADATA_PATH)
    macaque_class_names = get_class_names(MACAQUE_ADATA_PATH)

    summary_rows = []

    for case in SELECTED_CASES:
        har_id = case["har_id"]
        group = case["group"]

        print("\n" + "=" * 90)
        print(f"Processing {har_id} | {group}")

        for cfg in PLOT_CONFIGS:
            seq_species = cfg["seq_species"]
            model_label = cfg["model_label"]

            print("-" * 60)
            print(f"seq_species={seq_species} | model={model_label}")

            row = get_case_row(long_df, har_id, seq_species, model_label)
            if row is None:
                print(f"[SKIP] missing row for har_id={har_id}, seq_species={seq_species}, model={model_label}")
                continue

            region_str, seq = fetch_sequence(row)
            print("Region:", region_str)
            print("Sequence length:", len(seq))

            seq_start = int(row["seq_start"])
            seq_end = int(row["seq_end"])

            motif_hits = get_top_motif_hits(
                fimo_df=fimo_df,
                har_id=har_id,
                seq_species=seq_species,
                seq_start=seq_start,
                seq_end=seq_end,
                top_n=TOP_N_MOTIFS,
            )
            print(f"Top motif hits kept: {motif_hits.shape[0]}")

            out_prefix = OUT_DIR / f"{group}__{sanitize_name(har_id)}"

            target_topic = str(row["top_topic"]) if "top_topic" in row.index else str(row["target_topic"])

            if model_label == "human_model":
                model = human_model
                class_names = human_class_names
            elif model_label == "macaque_model":
                model = macaque_model
                class_names = macaque_class_names
            else:
                raise ValueError(f"Unsupported model_label: {model_label}")

            res = run_contribution_plot_with_manual_highlights(
                sequence=seq,
                row=row,
                motif_hits=motif_hits,
                har_id=har_id,
                group=group,
                seq_species=seq_species,
                region_str=region_str,
                model=model,
                model_label=model_label,
                target_topic=target_topic,
                class_names=class_names,
                out_prefix=out_prefix,
            )

            save_top5_table(
                motif_hits=motif_hits,
                out_prefix=out_prefix,
                seq_species=seq_species,
                model_label=model_label,
            )

            summary_rows.append({
                "group": group,
                "har_id": har_id,
                "seq_species": seq_species,
                "model_label": model_label,
                "region": region_str,
                "target_topic": target_topic,
                "top_score": float(row["top_score"]) if "top_score" in row.index else np.nan,
                "best_celltype_by_mean": str(row["best_celltype_by_mean"]) if "best_celltype_by_mean" in row.index else "",
                "n_top5_motifs_found": res["n_motif_hits"],
            })

    summary_df = pd.DataFrame(summary_rows)
    out_summary = OUT_DIR / "26_2_har_top5motif_manual_summary.csv"
    summary_df.to_csv(out_summary, index=False)

    print("\nSaved:", out_summary)
    print("Done.")


if __name__ == "__main__":
    main()