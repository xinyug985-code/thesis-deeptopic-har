#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from pathlib import Path
import re
import pandas as pd
import numpy as np


# ============================================================
# paths
# ============================================================
BASE = Path("/mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/pjt")

HUMAN_BED_DIR = BASE / "data" / "topic_beds_human_otsu"
MACAQUE_BED_DIR = BASE / "data" / "topic_beds_macaque_otsu"

HUMAN_TSV = BASE / "data" / "human_topic_top_celltype.tsv"
MACAQUE_TSV = BASE / "data" / "macaque_topic_top_celltype.tsv"

OUTDIR = BASE / "data" / "stage0_annotation"
OUTDIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# thresholds
# ============================================================
HIGH_RATIO = 1.5
MEDIUM_RATIO = 1.2


# ============================================================
# lineage mapping 
# ============================================================
HUMAN_LINEAGE_MAP = {
    "CGE_LGE_progenitors": "progenitor",
    "CGE_interneuron": "interneuron",
    "ExIPC": "progenitor",
    "ExNeu_IT": "excitatory_neuron",
    "ExNeu_Non_IT": "excitatory_neuron",
    "LGE_FOXP1_ISL1_MSN": "subpallial_neuron",
    "LGE_FOXP1_PENK_MSN": "subpallial_neuron",
    "LGE_FOXP2_TSHZ1_MSN": "subpallial_neuron",
    "LGE_OB_interneuron": "interneuron",
    "MGE_interneuron": "interneuron",
    "MGE_progenitors": "progenitor",
    "Microglia": "immune",
    "Neuroblast": "neuronal_intermediate",
    "OPC": "glia",
    "TriIPC_Astrocyte": "glia_like_progenitor",
    "Vascular": "vascular",
    "dorsal_RG": "progenitor",
}

MACAQUE_LINEAGE_MAP = {
    "Astrocyte": "glia",
    "CGE_interneuron": "interneuron",
    "ExNeu_IT": "excitatory_neuron",
    "ExNeu_Non_IT": "excitatory_neuron",
    "GPC": "glia_progenitor",
    "LGE_FOXP2_TSHZ1": "subpallial_neuron",
    "MGE_interneuron": "interneuron",
    "Microglia": "immune",
    "OPC": "glia",
    "Oligo": "glia",
    "Vascular": "vascular",
}


# ============================================================
# helpers
# ============================================================
def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")


def require_dir(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing directory: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Not a directory: {path}")


def normalize_topic(topic: str) -> str:
    s = str(topic).strip()
    m = re.search(r"topic[_\s-]*(\d+)", s, flags=re.IGNORECASE)
    if m:
        return f"Topic{int(m.group(1))}"
    return s


def topic_num(topic: str) -> int | None:
    m = re.search(r"Topic(\d+)", str(topic))
    if m:
        return int(m.group(1))
    return None


def classify_confidence(ratio: float) -> str:
    if pd.isna(ratio):
        return "unknown"
    if ratio >= HIGH_RATIO:
        return "high"
    if ratio >= MEDIUM_RATIO:
        return "medium"
    return "low"


def classify_annotation_class(ratio: float) -> str:
    if pd.isna(ratio):
        return "unknown"
    if ratio >= HIGH_RATIO:
        return "clean_top1"
    return "ambiguous"


def scan_topic_beds(bed_dir: Path) -> dict[str, Path]:
    mapping = {}
    bed_files = sorted(bed_dir.glob("*.bed"))

    for fp in bed_files:
        m = re.search(r"topic[_\s-]*(\d+)", fp.stem, flags=re.IGNORECASE)
        if m:
            topic = f"Topic{int(m.group(1))}"
            mapping[topic] = fp

    return mapping


def count_bed_rows(path: Path) -> int:
    n = 0
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                continue
            n += 1
    return n


def load_topic_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")

    required = [
        "topic",
        "top1_celltype",
        "top1_score",
        "top2_celltype",
        "top2_score",
        "specificity_ratio_top1_over_top2",
    ]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"{path} missing required column: {c}")

    df = df.copy()
    df["topic"] = df["topic"].astype(str).map(normalize_topic)
    df["top1_score"] = pd.to_numeric(df["top1_score"], errors="coerce")
    df["top2_score"] = pd.to_numeric(df["top2_score"], errors="coerce")
    df["specificity_ratio_top1_over_top2"] = pd.to_numeric(
        df["specificity_ratio_top1_over_top2"], errors="coerce"
    )
    return df


def build_topic_annotation(
    df: pd.DataFrame,
    species: str,
    bed_dir: Path,
    lineage_map: dict[str, str],
) -> pd.DataFrame:
    bed_map = scan_topic_beds(bed_dir)

    out = df.copy()
    out["species"] = species

    out["annotation_confidence"] = out["specificity_ratio_top1_over_top2"].map(classify_confidence)
    out["annotation_class"] = out["specificity_ratio_top1_over_top2"].map(classify_annotation_class)

    out["lineage_group"] = out["top1_celltype"].map(lineage_map)
    out["lineage_group_top2"] = out["top2_celltype"].map(lineage_map)

    out["bed_path"] = out["topic"].map(lambda x: str(bed_map[x]) if x in bed_map else "")
    out["bed_exists"] = out["topic"].map(lambda x: x in bed_map)

    n_peaks = []
    for _, row in out.iterrows():
        topic = row["topic"]
        if topic in bed_map:
            try:
                n = count_bed_rows(bed_map[topic])
            except Exception:
                n = np.nan
        else:
            n = np.nan
        n_peaks.append(n)
    out["n_peaks"] = n_peaks

    out["topic_num"] = out["topic"].map(topic_num)
    out["score_delta_top1_minus_top2"] = out["top1_score"] - out["top2_score"]

    out["usable_for_motif"] = (
        (out["annotation_confidence"] == "high")
        & (out["bed_exists"] == True)
        & (pd.to_numeric(out["n_peaks"], errors="coerce").fillna(0) > 0)
    )

    cols = [
        "topic",
        "topic_num",
        "species",
        "top1_celltype",
        "top1_score",
        "top2_celltype",
        "top2_score",
        "specificity_ratio_top1_over_top2",
        "score_delta_top1_minus_top2",
        "annotation_confidence",
        "annotation_class",
        "lineage_group",
        "lineage_group_top2",
        "bed_path",
        "bed_exists",
        "n_peaks",
        "usable_for_motif",
    ]

    out = out[cols].sort_values(["topic_num", "topic"]).reset_index(drop=True)
    return out


def build_celltype_summary(topic_anno: pd.DataFrame, species: str) -> pd.DataFrame:
    rows = []

    for celltype, sub in topic_anno.groupby("top1_celltype", dropna=False):
        topic_list = sub.sort_values(["topic_num", "topic"])["topic"].tolist()
        usable_topics = sub.loc[sub["usable_for_motif"], "topic"].tolist()

        row = {
            "species": species,
            "celltype": celltype,
            "lineage_group": sub["lineage_group"].mode().iloc[0] if sub["lineage_group"].notna().any() else np.nan,
            "n_topics_top1": int(sub.shape[0]),
            "n_topics_high_confidence": int((sub["annotation_confidence"] == "high").sum()),
            "n_topics_medium_or_high": int(sub["annotation_confidence"].isin(["high", "medium"]).sum()),
            "n_topics_usable_for_motif": int(sub["usable_for_motif"].sum()),
            "mean_top1_score": float(sub["top1_score"].mean()),
            "median_top1_score": float(sub["top1_score"].median()),
            "mean_specificity_ratio": float(sub["specificity_ratio_top1_over_top2"].mean()),
            "median_specificity_ratio": float(sub["specificity_ratio_top1_over_top2"].median()),
            "total_peaks_across_topics": float(pd.to_numeric(sub["n_peaks"], errors="coerce").fillna(0).sum()),
            "median_peaks_per_topic": float(pd.to_numeric(sub["n_peaks"], errors="coerce").median()),
            "topics_list": ",".join(topic_list),
            "usable_topics_list": ",".join(usable_topics),
        }
        rows.append(row)

    out = pd.DataFrame(rows)

    if not out.empty:
        out = out.sort_values(
            ["n_topics_top1", "n_topics_usable_for_motif", "mean_specificity_ratio"],
            ascending=[False, False, False]
        ).reset_index(drop=True)

    return out


def build_qc_table(topic_anno: pd.DataFrame, species: str) -> pd.DataFrame:
    metrics = []

    metrics.append({"species": species, "metric": "n_total_topics", "value": int(topic_anno.shape[0])})
    metrics.append({"species": species, "metric": "n_bed_exists", "value": int(topic_anno["bed_exists"].sum())})
    metrics.append({"species": species, "metric": "n_missing_bed", "value": int((~topic_anno["bed_exists"]).sum())})
    metrics.append({"species": species, "metric": "n_high_confidence", "value": int((topic_anno["annotation_confidence"] == "high").sum())})
    metrics.append({"species": species, "metric": "n_medium_confidence", "value": int((topic_anno["annotation_confidence"] == "medium").sum())})
    metrics.append({"species": species, "metric": "n_low_confidence", "value": int((topic_anno["annotation_confidence"] == "low").sum())})
    metrics.append({"species": species, "metric": "n_usable_for_motif", "value": int(topic_anno["usable_for_motif"].sum())})

    return pd.DataFrame(metrics)


def save_tsv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, sep="\t", index=False)


# ============================================================
# main
# ============================================================
def main():
    print("Checking inputs...")
    require_file(HUMAN_TSV)
    require_file(MACAQUE_TSV)
    require_dir(HUMAN_BED_DIR)
    require_dir(MACAQUE_BED_DIR)

    print("Loading topic tables...")
    human_df = load_topic_table(HUMAN_TSV)
    macaque_df = load_topic_table(MACAQUE_TSV)

    print("Building human annotation...")
    human_anno = build_topic_annotation(
        df=human_df,
        species="human",
        bed_dir=HUMAN_BED_DIR,
        lineage_map=HUMAN_LINEAGE_MAP,
    )

    print("Building macaque annotation...")
    macaque_anno = build_topic_annotation(
        df=macaque_df,
        species="macaque",
        bed_dir=MACAQUE_BED_DIR,
        lineage_map=MACAQUE_LINEAGE_MAP,
    )

    print("Building summaries...")
    human_summary = build_celltype_summary(human_anno, species="human")
    macaque_summary = build_celltype_summary(macaque_anno, species="macaque")

    human_qc = build_qc_table(human_anno, species="human")
    macaque_qc = build_qc_table(macaque_anno, species="macaque")

    human_missing_bed = human_anno.loc[~human_anno["bed_exists"]].copy()
    macaque_missing_bed = macaque_anno.loc[~macaque_anno["bed_exists"]].copy()

    human_anno_out = OUTDIR / "topic_annotation_human.tsv"
    macaque_anno_out = OUTDIR / "topic_annotation_macaque.tsv"

    human_summary_out = OUTDIR / "celltype_summary_human.tsv"
    macaque_summary_out = OUTDIR / "celltype_summary_macaque.tsv"

    human_qc_out = OUTDIR / "stage0_qc_human.tsv"
    macaque_qc_out = OUTDIR / "stage0_qc_macaque.tsv"

    human_missing_bed_out = OUTDIR / "missing_bed_topics_human.tsv"
    macaque_missing_bed_out = OUTDIR / "missing_bed_topics_macaque.tsv"

    save_tsv(human_anno, human_anno_out)
    save_tsv(macaque_anno, macaque_anno_out)
    save_tsv(human_summary, human_summary_out)
    save_tsv(macaque_summary, macaque_summary_out)
    save_tsv(human_qc, human_qc_out)
    save_tsv(macaque_qc, macaque_qc_out)
    save_tsv(human_missing_bed, human_missing_bed_out)
    save_tsv(macaque_missing_bed, macaque_missing_bed_out)

    print("\nSaved files:")
    print(human_anno_out)
    print(macaque_anno_out)
    print(human_summary_out)
    print(macaque_summary_out)
    print(human_qc_out)
    print(macaque_qc_out)
    print(human_missing_bed_out)
    print(macaque_missing_bed_out)

    print("\nQuick summary:")
    print(f"Human topics total:   {human_anno.shape[0]}")
    print(f"Human motif-usable:   {int(human_anno['usable_for_motif'].sum())}")
    print(f"Macaque topics total: {macaque_anno.shape[0]}")
    print(f"Macaque motif-usable: {int(macaque_anno['usable_for_motif'].sum())}")


if __name__ == "__main__":
    main()