#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse
import pickle
import numpy as np
import pandas as pd

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
        "pattern_summary": SUMMARY / "human_modisco_pattern_summary.tsv",
    },
    "macaque": {
        "modisco_dir": STAGE2 / "macaque" / "modisco_results_top1000",
        "pattern_summary": SUMMARY / "macaque_modisco_pattern_summary.tsv",
    },
}


def get_topic_classes(modisco_dir: Path):
    h5_files = sorted(modisco_dir.glob("*_modisco_results.h5"))
    classes = [f.name.replace("_modisco_results.h5", "") for f in h5_files]
    return classes


def select_top_topics(pattern_summary_tsv: Path, top_n=100):
    df = pd.read_csv(pattern_summary_tsv, sep="\t")

    sub = df[df["pattern_group"] == "pos_patterns"].copy()

    topic_rank = (
        sub.groupby("topic")["n_seqlets"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
    )

    return list(topic_rank.index)


def run_species(species: str, top_n_topics=100, min_seqlets=100):
    cfg = DATASETS[species]

    modisco_dir = cfg["modisco_dir"]
    pattern_summary = cfg["pattern_summary"]

    outdir = STAGE2 / species / "pattern_matching_tomtom"
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"===== {species} =====")
    print("[modisco_dir]", modisco_dir)
    print("[pattern_summary]", pattern_summary)

    classes = get_topic_classes(modisco_dir)
    print("[n classes/topics]", len(classes))
    print("[first classes]", classes[:5])

    # 1. match .h5 files
    matched_files = crested.tl.modisco.match_h5_files_to_classes(
        contribution_dir=str(modisco_dir),
        classes=classes,
    )

    with open(outdir / f"{species}_matched_modisco_files.pkl", "wb") as f:
        pickle.dump(matched_files, f)

    print("[matched files]", len(matched_files))

    # 2. TOMTOM similarity between each pattern
    sim_matrix, pattern_ids, pattern_dict = crested.tl.modisco.calculate_tomtom_similarity_per_pattern(
        matched_files=matched_files,
        trim_ic_threshold=0.025,
        verbose=True,
    )

    # save
    np.save(outdir / f"{species}_tomtom_similarity_matrix.npy", sim_matrix)

    pd.DataFrame(
        sim_matrix,
        index=pattern_ids,
        columns=pattern_ids,
    ).to_csv(outdir / f"{species}_tomtom_similarity_matrix.tsv", sep="\t")

    pd.Series(pattern_ids, name="pattern_id").to_csv(
        outdir / f"{species}_tomtom_pattern_ids.tsv",
        sep="\t",
        index=False,
    )

    with open(outdir / f"{species}_tomtom_pattern_dict.pkl", "wb") as f:
        pickle.dump(pattern_dict, f)

    print("[save]", outdir / f"{species}_tomtom_similarity_matrix.tsv")
    print("[n patterns]", len(pattern_ids))

    # 3. plot clustermap
    top_topics = select_top_topics(pattern_summary, top_n=top_n_topics)
    print("[top topics for plot]", top_topics)

    try:
        cg = crested.pl.modisco.clustermap_tomtom_similarities(
            sim_matrix=sim_matrix,
            ids=pattern_ids,
            pattern_dict=pattern_dict,
            class_names=top_topics,
            min_seqlets=min_seqlets,
        )

        cg.fig.savefig(
            outdir / f"{species}_tomtom_clustermap_top{top_n_topics}_topics_minseqlets{min_seqlets}.pdf",
            bbox_inches="tight",
        )
        cg.fig.savefig(
            outdir / f"{species}_tomtom_clustermap_top{top_n_topics}_topics_minseqlets{min_seqlets}.png",
            dpi=220,
            bbox_inches="tight",
        )
        plt.close(cg.fig)

        print("[save clustermap]", outdir)

    except Exception as e:
        print("[WARN] clustermap failed:", e)

    params = {
        "species": species,
        "n_classes": len(classes),
        "n_patterns": len(pattern_ids),
        "trim_ic_threshold": 0.025,
        "top_n_topics_for_plot": top_n_topics,
        "min_seqlets_for_plot": min_seqlets,
        "modisco_dir": str(modisco_dir),
        "outdir": str(outdir),
    }

    pd.Series(params).to_json(
        outdir / f"{species}_pattern_matching_params.json",
        indent=2,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--species",
        choices=["human", "macaque", "both"],
        default="both",
    )
    parser.add_argument(
        "--top-n-topics",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--min-seqlets",
        type=int,
        default=100,
    )

    args = parser.parse_args()

    if args.species == "both":
        for sp in ["human", "macaque"]:
            run_species(
                sp,
                top_n_topics=args.top_n_topics,
                min_seqlets=args.min_seqlets,
            )
    else:
        run_species(
            args.species,
            top_n_topics=args.top_n_topics,
            min_seqlets=args.min_seqlets,
        )


if __name__ == "__main__":
    main()