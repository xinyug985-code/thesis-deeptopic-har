#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 2: CREsted enhancer-code analysis for topic-classification models.

Main design:
    sequence -> 100 pycisTopic topics
    topic-level Gini region selection
    topic-level contribution_scores_specific
    topic-level TF-MoDISco-lite
    final interpretation: topic -> cell type annotation table

This follows the CREsted enhancer_code_analysis tutorial structure:
    1. load AnnData/model/genome
    2. predict all regions
    3. store prediction layer
    4. create combined = (X + prediction) / 2
    5. inspect sort_and_filter_cutoff
    6. sort_and_filter_regions_on_specificity(top_k=1000, method='gini')
    7. contribution_scores_specific(target_idx=None)
    8. tfmodisco
"""

# %% 01. Package settings and imports
import os
os.environ.setdefault("KERAS_BACKEND", "torch")

from pathlib import Path
import argparse
import json
import warnings

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
import crested
import keras

warnings.filterwarnings("ignore", category=FutureWarning)


# %% 02. User paths and global configuration
BASE = Path("/mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/pjt")
DATA = BASE / "data"
OUT = BASE / "runs" / "out"
STAGE2_OUT = OUT / "stage2_topic_modisco_top1000"
STAGE2_OUT.mkdir(parents=True, exist_ok=True)

TOP_K = 1000
METHOD = "gini"
MODEL_LAYER = "model_prediction"
COMBINED_LAYER = "combined"
CONTRIB_METHOD = "integrated_grad"
SEQ_WINDOW = 500
MAX_SEQLETS = 20000

DATASETS = {
    "human": {
        "adata": OUT / "human_topics.h5ad",
        "model": OUT / "deeptopic_human" / "final_model.keras",
        "genome_fasta": Path("/mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/genomes/hg38/hg38.fa"),
        "chrom_sizes": Path("/mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/genomes/hg38/hg38.chrom.sizes"),
        "annotation_candidates": DATA / "3_topic_annotation_human_pycistopic.tsv",
        "qc_candidates": DATA / "3_topic_qc_metrics_human.tsv",
    },
    "macaque": {
        "adata": OUT / "macaque_topics.h5ad",
        "model": OUT / "deeptopic_macaque" / "final_model.keras",
        "genome_fasta": Path("/mimer/NOBACKUP/groups/naiss2025-22-612/yiquan/macaque/rheMac10.fa"),
        "chrom_sizes": Path("/mimer/NOBACKUP/groups/naiss2025-22-612/yiquan/macaque/rheMac10.chrom.sizes"),
        "annotation_candidates": DATA / "3_topic_annotation_macaque_pycistopic.tsv",
        "qc_candidates": DATA / "3_topic_qc_metrics_macaque.tsv",
    },
}


# %% 03. Small helpers
def check_exists(path: Path, label: str) -> None:
    if not Path(path).exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def normalize_topic_name(x) -> str:
    x = str(x).strip()
    if x.startswith("Topic"):
        return x
    try:
        return f"Topic{int(float(x))}"
    except Exception:
        return x


def load_table(path: Path, label: str) -> pd.DataFrame:
    check_exists(path, label)
    print(f"[load {label}] {path}")
    if str(path).endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_csv(path, sep="\t")


def infer_topic_col(df: pd.DataFrame) -> str:
    for col in ["topic", "Topic", "topic_id", "TopicID", "class", "Class name"]:
        if col in df.columns:
            return col
    raise ValueError(f"Cannot infer topic column from columns: {list(df.columns)}")


def infer_celltype_col(df: pd.DataFrame) -> str | None:
    for col in ["final_class", "celltype", "cell_type", "annotation", "top_celltype", "best_celltype"]:
        if col in df.columns:
            return col
    return None


def load_model_safely(model_path: Path):
    check_exists(model_path, "model")
    print(f"[load model] {model_path}")
    try:
        return crested.utils.load_model(str(model_path))
    except Exception as e:
        print(f"[warn] crested.utils.load_model failed: {e}")
        print("[fallback] keras.models.load_model")
        return keras.models.load_model(model_path)


def ensure_prediction_layer(adata: ad.AnnData, model, layer_name: str) -> ad.AnnData:
    if layer_name in adata.layers:
        print(f"[predict] reuse adata.layers['{layer_name}']")
        return adata

    print("[predict] running crested.tl.predict with default batch size")
    pred = crested.tl.predict(adata, model)
    print("[pred shape]", pred.shape)
    print("[adata shape]", adata.shape)

    if pred.shape == adata.shape:
        adata.layers[layer_name] = pred
    elif pred.T.shape == adata.shape:
        adata.layers[layer_name] = pred.T
    else:
        raise ValueError(
            f"Prediction shape {pred.shape} cannot fit AnnData shape {adata.shape}. "
            "Expected either same shape or transpose shape."
        )
    return adata


def add_combined_layer(adata: ad.AnnData, pred_layer: str, combined_layer: str) -> ad.AnnData:
    print(f"[combined] {combined_layer} = (adata.X + {pred_layer}) / 2")
    adata.layers[combined_layer] = (np.asarray(adata.X) + np.asarray(adata.layers[pred_layer])) / 2
    return adata


def make_topic_annotation_map(annotation_df: pd.DataFrame) -> pd.DataFrame:
    topic_col = infer_topic_col(annotation_df)
    celltype_col = infer_celltype_col(annotation_df)

    out = annotation_df.copy()
    out["topic_norm"] = out[topic_col].map(normalize_topic_name)

    keep_cols = ["topic_norm"]
    if topic_col not in keep_cols:
        keep_cols.append(topic_col)
    if celltype_col is not None and celltype_col not in keep_cols:
        keep_cols.append(celltype_col)
    for col in ["is_general", "specificity", "specificity_ratio", "top_score", "top2_score"]:
        if col in out.columns and col not in keep_cols:
            keep_cols.append(col)

    out = out[keep_cols].drop_duplicates("topic_norm")
    return out


def save_filtered_region_table(adata_filtered: ad.AnnData, annotation_map: pd.DataFrame, out_tsv: Path) -> None:
    var = adata_filtered.var.copy()
    var = var.reset_index().rename(columns={"index": "region"})

    # CREsted adds the selected class name into var['Class name'] in the tutorial.
    if "Class name" in var.columns:
        var["topic_norm"] = var["Class name"].map(normalize_topic_name)
    elif "class_name" in var.columns:
        var["topic_norm"] = var["class_name"].map(normalize_topic_name)
    else:
        var["topic_norm"] = pd.NA

    merged = var.merge(annotation_map, on="topic_norm", how="left")
    merged.to_csv(out_tsv, sep="\t", index=False)
    print(f"[save] {out_tsv}")


def write_summary(path: Path, payload: dict) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"[save] {path}")


# %% 04. Main Stage 2 function for one species
def run_species(species: str) -> None:
    if species not in DATASETS:
        raise ValueError(f"Unknown species={species}. Use one of: {list(DATASETS)}")

    cfg = DATASETS[species]
    species_out = STAGE2_OUT / species
    species_out.mkdir(parents=True, exist_ok=True)
    contrib_dir = species_out / f"modisco_contrib_top{TOP_K}"
    modisco_dir = species_out / f"modisco_results_top{TOP_K}"
    contrib_dir.mkdir(parents=True, exist_ok=True)
    modisco_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n===== Stage 2 topic-level enhancer code analysis: {species} =====")
    print(f"[out] {species_out}")

    # 04.1 Load genome, AnnData, annotation, model
    check_exists(cfg["genome_fasta"], "genome_fasta")
    check_exists(cfg["chrom_sizes"], "chrom_sizes")
    genome = crested.Genome(str(cfg["genome_fasta"]), str(cfg["chrom_sizes"]))
    crested.register_genome(genome)

    check_exists(cfg["adata"], "adata")
    print(f"[load adata] {cfg['adata']}")
    adata = ad.read_h5ad(cfg["adata"])
    print("[adata]", adata)

    annot_df = load_table(cfg["annotation_candidates"], "annotation")
    annotation_map = make_topic_annotation_map(annot_df)
    annotation_map_path = species_out / f"{species}_topic_to_celltype_annotation_used.tsv"
    annotation_map.to_csv(annotation_map_path, sep="\t", index=False)
    print(f"[save] {annotation_map_path}")

    if cfg["qc_candidates"].exists():
        qc_df = load_table(cfg["qc_candidates"], "qc")
        qc_out = species_out / f"{species}_topic_qc_metrics_copy.tsv"
        qc_df.to_csv(qc_out, sep="\t", index=False)
        print(f"[save] {qc_out}")

    model = load_model_safely(cfg["model"])

    # 04.2 Predict all regions and create combined layer
    adata = ensure_prediction_layer(adata, model, MODEL_LAYER)
    adata = add_combined_layer(adata, MODEL_LAYER, COMBINED_LAYER)

    # Optional: save a copy with prediction/combined layers for restart.
    pred_h5ad = species_out / f"{species}_topics_with_prediction_combined.h5ad"
    adata.write_h5ad(pred_h5ad)
    print(f"[save] {pred_h5ad}")

    # 04.3 Tutorial-style cutoff plot: topic-level Gini per topic
    print("[plot] crested.pl.qc.sort_and_filter_cutoff")
    plt.figure()
    crested.pl.qc.sort_and_filter_cutoff(
        adata,
        model_name=COMBINED_LAYER,
        cutoffs=[500, 1000, 2000],
        max_k=5000,
    )
    cutoff_png = species_out / f"{species}_topic_gini_sort_and_filter_cutoff_top{TOP_K}.png"
    cutoff_pdf = species_out / f"{species}_topic_gini_sort_and_filter_cutoff_top{TOP_K}.pdf"
    plt.savefig(cutoff_png, dpi=220, bbox_inches="tight")
    plt.savefig(cutoff_pdf, bbox_inches="tight")
    plt.close("all")
    print(f"[save] {cutoff_png}")
    print(f"[save] {cutoff_pdf}")

    # 04.4 Filter most informative regions per topic
    print(f"[filter] sort_and_filter_regions_on_specificity top_k={TOP_K}, method={METHOD}, model_name={COMBINED_LAYER}")
    adata_filtered = crested.pp.sort_and_filter_regions_on_specificity(
        adata,
        model_name=COMBINED_LAYER,
        top_k=TOP_K,
        method=METHOD,
        inplace=False,
    )
    print("[adata_filtered]", adata_filtered)

    filtered_h5ad = species_out / f"{species}_topics_filtered_top{TOP_K}_{METHOD}.h5ad"
    adata_filtered.write_h5ad(filtered_h5ad)
    print(f"[save] {filtered_h5ad}")

    filtered_regions_tsv = species_out / f"{species}_filtered_regions_top{TOP_K}_{METHOD}_with_topic_annotation.tsv"
    save_filtered_region_table(adata_filtered, annotation_map, filtered_regions_tsv)

    # 04.5 Calculate contribution scores for all topic classes
    print("[contribution] crested.tl.contribution_scores_specific")
    print(f"[contribution output_dir] {contrib_dir}")
    crested.tl.contribution_scores_specific(
        input=adata_filtered,
        target_idx=None,
        model=model,
        output_dir=str(contrib_dir),
        method=CONTRIB_METHOD,
    )

    # 04.6 Run TF-MoDISco-lite on saved contribution scores
    print("[tfmodisco] crested.tl.modisco.tfmodisco")
    print(f"[modisco output_dir] {modisco_dir}")
    crested.tl.modisco.tfmodisco(
        window=SEQ_WINDOW,
        output_dir=str(modisco_dir),
        contrib_dir=str(contrib_dir),
        report=False,
        max_seqlets=MAX_SEQLETS,
    )

    # 04.7 Save run summary
    summary = {
        "species": species,
        "top_k": TOP_K,
        "method": METHOD,
        "model_layer": MODEL_LAYER,
        "combined_layer": COMBINED_LAYER,
        "contrib_method": CONTRIB_METHOD,
        "seq_window": SEQ_WINDOW,
        "max_seqlets": MAX_SEQLETS,
        "adata": cfg["adata"],
        "model": cfg["model"],
        "genome_fasta": cfg["genome_fasta"],
        "chrom_sizes": cfg["chrom_sizes"],
        "annotation": cfg["annotation_candidates"],
        "prediction_h5ad": pred_h5ad,
        "filtered_h5ad": filtered_h5ad,
        "filtered_regions_tsv": filtered_regions_tsv,
        "contribution_dir": contrib_dir,
        "modisco_dir": modisco_dir,
    }
    write_summary(species_out / f"{species}_stage2_summary.json", summary)

    print(f"===== Done: {species} =====\n")


# %% 05. Command-line entry point for sbatch
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Stage 2 CREsted topic-level enhancer-code analysis.")
    parser.add_argument(
        "--species",
        choices=["human", "macaque", "both"],
        default="human",
        help="Species to run. Use 'both' to run human then macaque in one job.",
    )
    args = parser.parse_args()

    if args.species == "both":
        for sp in ["human", "macaque"]:
            run_species(sp)
    else:
        run_species(args.species)
