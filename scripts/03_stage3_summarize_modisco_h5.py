#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import re
import h5py
import numpy as np
import pandas as pd


BASE = Path("/mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/pjt")
OUT = BASE / "runs" / "out" / "stage2_topic_modisco_top1000"
SUMMARY_DIR = OUT / "summary"
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "human": {
        "modisco_dir": OUT / "human" / "modisco_results_top1000",
        "annotation": OUT / "human" / "human_topic_to_celltype_annotation_used.tsv",
    },
    "macaque": {
        "modisco_dir": OUT / "macaque" / "modisco_results_top1000",
        "annotation": OUT / "macaque" / "macaque_topic_to_celltype_annotation_used.tsv",
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


def topic_from_filename(path: Path) -> str:
    m = re.search(r"(Topic\d+)_modisco_results\.h5", path.name)
    if not m:
        return path.stem.replace("_modisco_results", "")
    return m.group(1)


def load_annotation(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    if "topic" not in df.columns:
        raise ValueError(f"No topic column in {path}")

    df["topic_norm"] = df["topic"].apply(normalize_topic_name)
    if "annotation" in df.columns:
        df["final_class"] = df["annotation"]

    keep_cols = ["topic_norm"]

    for c in ["final_class", "is_general"]:
        if c in df.columns:
            keep_cols.append(c)

    return df[keep_cols].drop_duplicates("topic_norm")


def safe_n_seqlets(g):
    if "seqlets" not in g:
        return 0
    sg = g["seqlets"]
    if "n_seqlets" in sg:
        return int(np.asarray(sg["n_seqlets"])[0])
    if "sequence" in sg:
        return int(sg["sequence"].shape[0])
    return 0


def pattern_stats(g):
    seq = np.asarray(g["sequence"]) if "sequence" in g else None
    contrib = np.asarray(g["contrib_scores"]) if "contrib_scores" in g else None
    hypo = np.asarray(g["hypothetical_contribs"]) if "hypothetical_contribs" in g else None

    length = seq.shape[0] if seq is not None else np.nan
    n_seqlets = safe_n_seqlets(g)

    contrib_abs_sum = float(np.abs(contrib).sum()) if contrib is not None else np.nan
    contrib_sum = float(contrib.sum()) if contrib is not None else np.nan
    max_abs_contrib = float(np.abs(contrib).max()) if contrib is not None else np.nan

    hypo_abs_sum = float(np.abs(hypo).sum()) if hypo is not None else np.nan

    return {
        "pattern_length": length,
        "n_seqlets": n_seqlets,
        "contrib_abs_sum": contrib_abs_sum,
        "contrib_sum": contrib_sum,
        "max_abs_contrib": max_abs_contrib,
        "hypothetical_abs_sum": hypo_abs_sum,
    }


def summarize_one_h5(h5_path: Path, species: str):
    topic = topic_from_filename(h5_path)
    rows = []

    with h5py.File(h5_path, "r") as h:
        for group_name in ["pos_patterns", "neg_patterns"]:
            if group_name not in h:
                continue

            group = h[group_name]
            pattern_names = sorted(
                [k for k in group.keys() if re.fullmatch(r"pattern_\d+", k)],
                key=lambda x: int(x.split("_")[1]),
            )

            for pname in pattern_names:
                g = group[pname]
                stats = pattern_stats(g)

                rows.append({
                    "species": species,
                    "topic": topic,
                    "pattern_group": group_name,
                    "pattern_name": pname,
                    "pattern_id": f"{species}_{topic}_{group_name}_{pname}",
                    "h5_file": str(h5_path),
                    **stats,
                })

    return rows


def summarize_species(species: str):
    cfg = DATASETS[species]
    modisco_dir = cfg["modisco_dir"]

    h5_files = sorted(modisco_dir.glob("*.h5"))
    print(f"[{species}] h5 files:", len(h5_files))

    all_rows = []
    for f in h5_files:
        all_rows.extend(summarize_one_h5(f, species))

    df = pd.DataFrame(all_rows)

    ann = load_annotation(cfg["annotation"])
    df["topic_norm"] = df["topic"].apply(normalize_topic_name)
    df = df.merge(ann, on="topic_norm", how="left")

    out_tsv = SUMMARY_DIR / f"{species}_modisco_pattern_summary.tsv"
    df.to_csv(out_tsv, sep="\t", index=False)

    print(f"[save] {out_tsv}")
    print(df.shape)

    if not df.empty:
        show_cols = [c for c in ["species", "topic", "pattern_group", "pattern_name", "n_seqlets", "final_class"] 
        if c in df.columns]
        print(df[show_cols].head())

    return df


def main():
    dfs = []
    for species in ["human", "macaque"]:
        dfs.append(summarize_species(species))

    all_df = pd.concat(dfs, ignore_index=True)
    out_all = SUMMARY_DIR / "all_species_modisco_pattern_summary.tsv"
    all_df.to_csv(out_all, sep="\t", index=False)
    print(f"[save] {out_all}")
    print(all_df.shape)


if __name__ == "__main__":
    main()