#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse
import pickle
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import crested


BASE = Path("/mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/pjt")
STAGE2 = BASE / "runs" / "out" / "stage2_topic_modisco_top1000"
SUMMARY = STAGE2 / "summary"

DATASETS = {
    "human": {
        "modisco_dir": STAGE2 / "human" / "modisco_results_top1000",
        "annotation": STAGE2 / "human" / "human_topic_to_celltype_annotation_used.tsv",
    },
    "macaque": {
        "modisco_dir": STAGE2 / "macaque" / "modisco_results_top1000",
        "annotation": STAGE2 / "macaque" / "macaque_topic_to_celltype_annotation_used.tsv",
    },
}


def normalize_topic_name(x):
    x = str(x).strip()
    if x.startswith("Topic"):
        return x
    try:
        return f"Topic{int(float(x))}"
    except Exception:
        return x


def load_annotation(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df["topic_norm"] = df["topic"].apply(normalize_topic_name)

    if "annotation" in df.columns:
        df["final_class"] = df["annotation"]

    keep = ["topic_norm", "final_class"]
    if "is_general" in df.columns:
        keep.append("is_general")

    return df[keep].drop_duplicates("topic_norm")


def get_topic_classes(modisco_dir: Path):
    h5_files = sorted(modisco_dir.glob("*_modisco_results.h5"))
    classes = [f.name.replace("_modisco_results.h5", "") for f in h5_files]
    classes = sorted(classes, key=lambda x: int(x.replace("Topic", "")) if x.replace("Topic", "").isdigit() else x)
    return classes


def run_species(species: str):
    cfg = DATASETS[species]
    modisco_dir = cfg["modisco_dir"]

    outdir = STAGE2 / species / "pattern_matrix"
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"===== {species} =====")
    print("[modisco_dir]", modisco_dir)

    classes = get_topic_classes(modisco_dir)
    print("[n classes]", len(classes))
    print("[classes head]", classes[:5])

    matched_files = crested.tl.modisco.match_h5_files_to_classes(
        contribution_dir=str(modisco_dir),
        classes=classes,
    )

    print("[process_patterns]")
    all_patterns = crested.tl.modisco.process_patterns(
        matched_files,
        sim_threshold=6.5,
        trim_ic_threshold=0.05,
        discard_ic_threshold=0.2,
        verbose=True,
    )

    with open(outdir / f"{species}_all_patterns.pkl", "wb") as f:
        pickle.dump(all_patterns, f)

    print("[create_pattern_matrix]")
    pattern_matrix = crested.tl.modisco.create_pattern_matrix(
        classes=classes,
        all_patterns=all_patterns,
        normalize=False,
        pattern_parameter="seqlet_count_log",
    )

    print("[pattern_matrix shape]", pattern_matrix.shape)

    pattern_df = pd.DataFrame(
        pattern_matrix,
        index=classes,
        columns=[f"pattern_cluster_{i}" for i in range(pattern_matrix.shape[1])],
    )
    pattern_df.index.name = "topic"

    pattern_df.to_csv(outdir / f"{species}_topic_by_pattern_matrix.tsv", sep="\t")

    ann = load_annotation(cfg["annotation"])
    pattern_df2 = pattern_df.reset_index()
    pattern_df2["topic_norm"] = pattern_df2["topic"].apply(normalize_topic_name)
    pattern_df2 = pattern_df2.merge(ann, on="topic_norm", how="left")
    pattern_df2.to_csv(outdir / f"{species}_topic_by_pattern_matrix_with_annotation.tsv", sep="\t", index=False)

    print("[save matrix]", outdir)

    print("[plot clustermap]")
    pat_seqs = crested.tl.modisco.generate_nucleotide_sequences(all_patterns)

    cg = crested.pl.modisco.clustermap(
        pattern_matrix,
        classes,
        width=30,
        height=10,
        pat_seqs=pat_seqs,
        grid=True,
        dendrogram_ratio=(0.03, 0.15),
        importance_threshold=5,
    )
    cg.fig.savefig(outdir / f"{species}_topic_pattern_clustermap.pdf", bbox_inches="tight")
    cg.fig.savefig(outdir / f"{species}_topic_pattern_clustermap.png", dpi=220, bbox_inches="tight")
    plt.close(cg.fig)

    try:
        cg2 = crested.pl.modisco.clustermap_with_pwm_logos(
            pattern_matrix,
            classes,
            pattern_dict=all_patterns,
            width=45,
            height=10,
            grid=True,
            dendrogram_ratio=(0.03, 0.15),
            importance_threshold=5,
            logo_x_multiplier=1,
            logo_height_fraction=0.35,
            logo_y_padding=0.25,
        )
        cg2.fig.canvas.draw()
        cg2.fig.savefig(outdir / f"{species}_topic_pattern_clustermap_with_pwm_logos.png", dpi=220, bbox_inches="tight")
        plt.close(cg2.fig)
    except Exception as e:
        print("[WARN] clustermap_with_pwm_logos failed:", e)

    print("[done]", species)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--species", choices=["human", "macaque", "both"], default="both")
    args = parser.parse_args()

    if args.species == "both":
        for sp in ["human", "macaque"]:
            run_species(sp)
    else:
        run_species(args.species)


if __name__ == "__main__":
    main()