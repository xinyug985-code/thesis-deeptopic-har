#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
os.environ["KERAS_BACKEND"] = "torch"

from pathlib import Path
import argparse
import re
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import anndata as ad
import crested
import keras
import torch


# ============================================================
# config
# ============================================================
BASE = Path("/mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/pjt")
OUT_DIR = BASE / "runs/out/selected_har_cases_deeptopic"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_LEN = 500

# beds
HUMAN_BED = BASE / "data/active_HARs_qc_pass.bed"
CHIMP_BED_CANDIDATES = [
    BASE / "data/active_HARs_pantro6_qc_pass.bed",
    BASE / "data/pantro6_active_HARs_qc_pass.bed",
    BASE / "data/chimp_active_HARs_qc_pass.bed",
]
MACAQUE_BED = BASE / "data/output_rheMac10.bed"

# model / adata
HUMAN_MODEL_PATH = BASE / "runs/out/deeptopic_human/final_model.keras"
HUMAN_ADATA_PATH = BASE / "runs/out/human_topics.h5ad"

# genome
HUMAN_FA = Path("/mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/genomes/hg38/hg38.fa")
HUMAN_CHROMSIZES = Path("/mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/genomes/hg38/hg38.chrom.sizes")

# topic annotation
HUMAN_TOPIC_ANNOT = BASE / "data/stage0_annotation/topic_annotation_human.tsv"

# selected cases
CASE_ALIASES = {
    "HARsv2_2749": ["HARsv2_2749"],
    "HARsv2_0583": ["HARsv2_0583"],
    "HARE5": ["HARsv2_0258"],
    "HAR1": ["HARsv2_1687"],
}


# ============================================================
# helpers
# ============================================================
def natural_topic_sort(cols):
    def key(x):
        m = re.match(r"Topic(\d+)$", str(x))
        return int(m.group(1)) if m else 10**9
    return sorted(cols, key=key)


def normalize_topic(x):
    x = str(x).strip()
    if x.startswith("Topic"):
        return x
    try:
        return f"Topic{int(float(x))}"
    except Exception:
        return x


def split_celltypes(x):
    if pd.isna(x):
        return []
    return [i.strip() for i in str(x).split(",") if i.strip()]


def load_topic_to_celltypes(annot_file):
    annot = pd.read_csv(annot_file, sep="\t")

    topic_col = "topic" if "topic" in annot.columns else annot.columns[0]

    if "final_class" in annot.columns:
        ct_col = "final_class"
    elif "celltype" in annot.columns:
        ct_col = "celltype"
    elif "top1_celltype" in annot.columns:
        ct_col = "top1_celltype"
    else:
        raise ValueError(f"Cannot find celltype column in {annot_file}: {annot.columns.tolist()}")

    topic_to_ct = {}

    for _, r in annot.iterrows():
        topic = normalize_topic(r[topic_col])

        if "is_general" in annot.columns:
            if str(r["is_general"]).lower() == "true":
                continue

        cts = split_celltypes(r[ct_col])
        if cts:
            topic_to_ct[topic] = cts

    return topic_to_ct


def read_bed_any(path):
    if not path.exists():
        return None

    df = pd.read_csv(path, sep="\t", header=None, comment="#")

    # 最少 BED3
    cols = ["chrom", "start", "end"]
    extra = [f"col{i}" for i in range(4, df.shape[1] + 1)]
    df.columns = cols + extra

    # 找 name 列
    if "col4" in df.columns:
        df["name"] = df["col4"].astype(str)
    else:
        df["name"] = df.index.astype(str)

    return df


def find_bed_row(df, aliases):
    if df is None or df.empty:
        return None

    names = df["name"].astype(str)

    for a in aliases:
        hit = df[names == a]
        if not hit.empty:
            return hit.iloc[0]

    for a in aliases:
        hit = df[names.str.contains(a, regex=False, na=False)]
        if not hit.empty:
            return hit.iloc[0]

    return None


def load_chromsizes(path):
    d = {}
    with open(path) as f:
        for line in f:
            chrom, size = line.rstrip("\n").split("\t")[:2]
            d[chrom] = int(size)
    return d


def center_window(chrom, start, end, chromsizes, length=500):
    mid = (int(start) + int(end)) // 2
    s = mid - length // 2
    e = s + length

    if s < 0:
        s = 0
        e = length

    if chrom in chromsizes and e > chromsizes[chrom]:
        e = chromsizes[chrom]
        s = max(0, e - length)

    return chrom, s, e


def fetch_sequence(genome, chrom, start, end):
    seq = genome.fetch(chrom, int(start), int(end)).upper()
    if len(seq) != TARGET_LEN:
        print(f"[WARN] sequence length {len(seq)} != {TARGET_LEN} for {chrom}:{start}-{end}")
    return seq


def one_hot_encode(seq):
    mapping = {
        "A": [1, 0, 0, 0],
        "C": [0, 1, 0, 0],
        "G": [0, 0, 1, 0],
        "T": [0, 0, 0, 1],
    }

    arr = np.zeros((len(seq), 4), dtype=np.float32)
    for i, b in enumerate(seq.upper()):
        arr[i] = mapping.get(b, [0.25, 0.25, 0.25, 0.25])
    return arr


def softmax_if_needed(pred):
    pred = np.asarray(pred, dtype=float)
    row_sum = pred.sum(axis=1)

    # 如果不像概率，就 softmax
    if np.any(pred < 0) or not np.allclose(row_sum, 1.0, atol=1e-2):
        x = pred - pred.max(axis=1, keepdims=True)
        exp = np.exp(x)
        pred = exp / exp.sum(axis=1, keepdims=True)

    return pred


def build_celltype_scores(score_df, topic_cols, topic_to_ct, agg="mean"):
    all_cts = sorted(set(ct for cts in topic_to_ct.values() for ct in cts))

    out = pd.DataFrame(index=score_df.index)

    for ct in all_cts:
        cols = [t for t in topic_cols if ct in topic_to_ct.get(t, [])]
        if not cols:
            continue

        if agg == "max":
            out[ct] = score_df[cols].max(axis=1)
        else:
            out[ct] = score_df[cols].mean(axis=1)

    return out


def row_zscore(df):
    arr = df.to_numpy(dtype=float)
    m = np.nanmean(arr, axis=1, keepdims=True)
    s = np.nanstd(arr, axis=1, keepdims=True)
    s[s == 0] = 1
    z = (arr - m) / s
    return pd.DataFrame(z, index=df.index, columns=df.columns)


def plot_heatmap(df, out_png, title, zscore=True):
    mat = row_zscore(df) if zscore else df.copy()

    fig_w = max(10, 0.35 * mat.shape[1])
    fig_h = max(4, 0.45 * mat.shape[0])

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    vmax = 2.5 if zscore else None

    im = ax.imshow(
        mat.to_numpy(dtype=float),
        aspect="auto",
        interpolation="nearest",
        vmin=-vmax if zscore else None,
        vmax=vmax,
    )

    ax.set_title(title)
    ax.set_xticks(np.arange(mat.shape[1]))
    ax.set_xticklabels(mat.columns, rotation=90, fontsize=7)
    ax.set_yticks(np.arange(mat.shape[0]))
    ax.set_yticklabels(mat.index, fontsize=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("row z-score" if zscore else "model score")

    fig.tight_layout()

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    print("[save]", out_png)
    print("[save]", out_png.with_suffix(".pdf"))

def plot_human_chimp_delta_heatmap(score_df, meta_df, out_png, title):
    """
    Rows = HAR cases
    Columns = cell types or topics
    Values = human score - chimp score
    """

    rows = []

    for case in sorted(meta_df["case"].unique()):
        human_id = f"{case}|human"
        chimp_id = f"{case}|chimp"

        if human_id not in score_df.index or chimp_id not in score_df.index:
            print(f"[SKIP delta] missing human/chimp pair for {case}")
            continue

        delta = score_df.loc[human_id] - score_df.loc[chimp_id]
        delta.name = case
        rows.append(delta)

    if len(rows) == 0:
        print("[WARN] no paired human/chimp cases for delta heatmap")
        return None

    delta_df = pd.DataFrame(rows)

    out_tsv = out_png.with_suffix(".tsv")
    delta_df.to_csv(out_tsv, sep="\t")
    print("[save delta table]", out_tsv)

    vmax = np.nanmax(np.abs(delta_df.to_numpy(dtype=float)))
    if vmax == 0 or np.isnan(vmax):
        vmax = None

    fig_w = max(10, 0.35 * delta_df.shape[1])
    fig_h = max(3.5, 0.6 * delta_df.shape[0])

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(
        delta_df.to_numpy(dtype=float),
        aspect="auto",
        interpolation="nearest",
        cmap="coolwarm",
        vmin=-vmax if vmax else None,
        vmax=vmax,
    )

    ax.set_title(title)
    ax.set_xticks(np.arange(delta_df.shape[1]))
    ax.set_xticklabels(delta_df.columns, rotation=90, fontsize=7)
    ax.set_yticks(np.arange(delta_df.shape[0]))
    ax.set_yticklabels(delta_df.index, fontsize=9)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("human - chimp score")

    fig.tight_layout()

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    print("[save]", out_png)
    print("[save]", out_png.with_suffix(".pdf"))

    return delta_df

# ============================================================
# main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agg", choices=["mean", "max"], default="mean")
    parser.add_argument("--zscore", action="store_true")
    args = parser.parse_args()

    print("[load adata]")
    adata = ad.read_h5ad(HUMAN_ADATA_PATH)
    topic_cols = natural_topic_sort(list(adata.obs_names))
    print("[n topics]", len(topic_cols))

    print("[load model]")
    model = keras.models.load_model(HUMAN_MODEL_PATH)

    print("[register genome]")
    genome = crested.Genome(str(HUMAN_FA), str(HUMAN_CHROMSIZES))
    crested.register_genome(genome)

    chromsizes = load_chromsizes(HUMAN_CHROMSIZES)

    print("[load bed]")
    human_bed = read_bed_any(HUMAN_BED)

    chimp_bed = None
    for p in CHIMP_BED_CANDIDATES:
        chimp_bed = read_bed_any(p)
        if chimp_bed is not None:
            print("[chimp bed]", p)
            break

    if chimp_bed is None:
        print("[WARN] no chimp bed found, only human reference cases will be scored")

    rows = []
    seq_records = []

    for case_name, aliases in CASE_ALIASES.items():
        print(f"\n===== {case_name} =====")

        for species, bed in [("human", human_bed), ("chimp", chimp_bed)]:
            if bed is None:
                continue

            r = find_bed_row(bed, aliases)

            if r is None:
                print(f"[MISS] {case_name} {species} aliases={aliases}")
                continue

            chrom, s, e = center_window(
                r["chrom"],
                int(r["start"]),
                int(r["end"]),
                chromsizes,
                length=TARGET_LEN,
            )

            seq = fetch_sequence(genome, chrom, s, e)

            seq_records.append({
                "case": case_name,
                "species": species,
                "record_id": f"{case_name}|{species}",
                "chrom": chrom,
                "start": s,
                "end": e,
                "original_start": int(r["start"]),
                "original_end": int(r["end"]),
                "sequence": seq,
            })

    if not seq_records:
        raise RuntimeError("No case sequences found. Check BED names/aliases.")

    print("\n[n sequences]", len(seq_records))

    X = np.stack([one_hot_encode(r["sequence"]) for r in seq_records], axis=0)

    print("[predict]", X.shape)
    pred = model.predict(X, batch_size=16, verbose=1)
    pred = softmax_if_needed(pred)

    score_df = pd.DataFrame(
        pred,
        columns=topic_cols,
        index=[r["record_id"] for r in seq_records],
    )

    meta_df = pd.DataFrame(seq_records).set_index("record_id")

    full_topic = pd.concat([meta_df, score_df], axis=1)
    full_topic["top_topic"] = score_df.idxmax(axis=1)
    full_topic["top_score"] = score_df.max(axis=1)
    full_topic["second_score"] = np.sort(score_df.to_numpy(), axis=1)[:, -2]
    full_topic["margin_top1_top2"] = full_topic["top_score"] - full_topic["second_score"]
    full_topic["entropy"] = -np.sum(score_df.to_numpy() * np.log(score_df.to_numpy() + 1e-9), axis=1)

    out_topic = OUT_DIR / "selected_cases_humanModel_topic_scores.tsv"
    full_topic.to_csv(out_topic, sep="\t")
    print("[save]", out_topic)

    topic_to_ct = load_topic_to_celltypes(HUMAN_TOPIC_ANNOT)
    ct_df = build_celltype_scores(score_df, topic_cols, topic_to_ct, agg=args.agg)

    full_ct = pd.concat([meta_df, ct_df], axis=1)
    full_ct["top_celltype"] = ct_df.idxmax(axis=1)
    full_ct["top_celltype_score"] = ct_df.max(axis=1)

    out_ct = OUT_DIR / f"selected_cases_humanModel_celltype_scores_{args.agg}.tsv"
    full_ct.to_csv(out_ct, sep="\t")
    print("[save]", out_ct)

    # summaries
    summary = meta_df[["case", "species", "chrom", "start", "end"]].copy()
    summary["top_topic"] = full_topic["top_topic"]
    summary["top_topic_score"] = full_topic["top_score"]
    summary["margin_top1_top2"] = full_topic["margin_top1_top2"]
    summary["entropy"] = full_topic["entropy"]
    summary["top_celltype"] = full_ct["top_celltype"]
    summary["top_celltype_score"] = full_ct["top_celltype_score"]

    out_sum = OUT_DIR / f"selected_cases_humanModel_summary_{args.agg}.tsv"
    summary.to_csv(out_sum, sep="\t")
    print("[save]", out_sum)

    # heatmaps
    plot_heatmap(
        score_df,
        OUT_DIR / "selected_cases_humanModel_topic_heatmap_rowZ.png",
        "Selected HAR cases × human DeepTopic topics",
        zscore=True,
    )

    plot_heatmap(
        ct_df,
        OUT_DIR / f"selected_cases_humanModel_celltype_heatmap_{args.agg}_rowZ.png",
        f"Selected HAR cases × cell types ({args.agg})",
        zscore=True,
    )

    plot_heatmap(
        ct_df,
        OUT_DIR / f"selected_cases_humanModel_celltype_heatmap_{args.agg}_raw.png",
        f"Selected HAR cases × cell types ({args.agg}, raw score)",
        zscore=False,
    )

        # delta heatmap: human - chimp
    plot_human_chimp_delta_heatmap(
        ct_df,
        meta_df,
        OUT_DIR / f"selected_cases_humanModel_celltype_delta_human_minus_chimp_{args.agg}.png",
        f"Selected HAR cases × cell types ({args.agg}, human - chimp)",
    )

    plot_human_chimp_delta_heatmap(
        score_df,
        meta_df,
        OUT_DIR / "selected_cases_humanModel_topic_delta_human_minus_chimp.png",
        "Selected HAR cases × topics (human - chimp)",
    )

    print("\nDone.")


if __name__ == "__main__":
    main()