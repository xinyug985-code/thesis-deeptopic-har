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

LONG_CSV = OUT_BASE / "master_har_3species_2models_long.csv"

HUMAN_MODEL_PATH = BASE_DIR / "runs" / "out" / "deeptopic_human" / "final_model.keras"
MACAQUE_MODEL_PATH = BASE_DIR / "runs" / "out" / "deeptopic_macaque" / "final_model.keras"

HUMAN_ADATA_PATH = BASE_DIR / "runs" / "out" / "human_topics.h5ad"
MACAQUE_ADATA_PATH = BASE_DIR / "runs" / "out" / "macaque_topics.h5ad"

HUMAN_FA = Path("/mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/genomes/hg38/hg38.fa")
MACAQUE_FA = Path("/mimer/NOBACKUP/groups/naiss2025-22-612/yiquan/macaque/rheMac10.fa")

OUT_DIR = BASE_DIR / "runs" / "out" / "case6_contribution_crossseq"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SELECTED_CASES = [
    {"har_id": "HARsv2_2699", "group": "human_high"},
    {"har_id": "HARsv2_1082", "group": "human_high"},
    {"har_id": "HARsv2_0513", "group": "human_high"},
    {"har_id": "HARsv2_0920", "group": "macaque_high"},
    {"har_id": "HARsv2_1547", "group": "macaque_high"},
    {"har_id": "HARsv2_0558", "group": "conserved"},
]

TARGET_SPECIES = ["human", "macaque"]
ZOOM_N_BASES = 500
FIG_DPI = 300


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


def run_contribution_and_plot(
    sequence: str,
    har_id: str,
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

    plt.figure(figsize=(14, 4))
    crested.pl.patterns.contribution_scores(
        scores,
        onehot,
        sequence_labels=[f"{har_id}|{seq_species}"],
        class_labels=[target_topic],
        zoom_n_bases=ZOOM_N_BASES,
        title=f"{har_id} | {seq_species} seq | {model_label} | {target_topic}",
        height=3,
    )
    plt.tight_layout()
    out_png = str(out_prefix) + f"__{seq_species}_seq__{model_label}__motif.png"
    out_pdf = str(out_prefix) + f"__{seq_species}_seq__{model_label}__motif.pdf"
    plt.savefig(out_png, dpi=FIG_DPI)
    plt.savefig(out_pdf)
    plt.close()
    print("Saved:", out_png)
    print("Saved:", out_pdf)

    # profile 
    arr_plot = extract_plot_matrix(scores)
    profile_abs, profile_signed = compute_profile(arr_plot)

    prof_df = pd.DataFrame({
        "position": np.arange(len(profile_abs)),
        "abs_contribution": profile_abs,
        "signed_contribution": profile_signed,
    })
    prof_csv = str(out_prefix) + f"__{seq_species}_seq__{model_label}__profile.csv"
    prof_df.to_csv(prof_csv, index=False)
    print("Saved:", prof_csv)

    plt.figure(figsize=(12, 3.5))
    plt.plot(profile_abs, label="abs contribution")
    plt.plot(profile_signed, label="signed contribution")
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("Position")
    plt.ylabel("Contribution")
    plt.title(f"{har_id} | {seq_species} seq | {model_label} | {target_topic}")
    plt.legend(frameon=False)
    out_png2 = str(out_prefix) + f"__{seq_species}_seq__{model_label}__profile.png"
    out_pdf2 = str(out_prefix) + f"__{seq_species}_seq__{model_label}__profile.pdf"
    plt.savefig(out_png2, dpi=FIG_DPI, bbox_inches="tight")
    plt.savefig(out_pdf2, bbox_inches="tight")
    plt.close()
    print("Saved:", out_png2)
    print("Saved:", out_pdf2)

    torch.cuda.empty_cache()
    gc.collect()

    return {
        "target_topic": target_topic,
        "target_idx": target_idx,
        "profile_abs": profile_abs,
        "profile_signed": profile_signed,
        "max_abs_contrib": float(np.max(profile_abs)),
        "sum_abs_contrib": float(np.sum(profile_abs)),
    }


def save_compare_plot(
    har_id: str,
    seq_species: str,
    region_str: str,
    vec_h: np.ndarray,
    vec_m: np.ndarray,
    out_prefix: Path,
):
    plt.figure(figsize=(12, 4))
    plt.plot(vec_h, label="human_model")
    plt.plot(vec_m, label="macaque_model")
    plt.xlabel("Position")
    plt.ylabel("Absolute contribution")
    plt.title(f"{har_id} | {seq_species} sequence | {region_str}")
    plt.legend(frameon=False)
    out_png = str(out_prefix) + f"__{seq_species}_seq__compare_abs_profile.png"
    out_pdf = str(out_prefix) + f"__{seq_species}_seq__compare_abs_profile.pdf"
    plt.savefig(out_png, dpi=FIG_DPI, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    print("Saved:", out_png)
    print("Saved:", out_pdf)


# ============================================================
# main
# ============================================================
def main():
    print("========== Cross-sequence contribution plots for selected 6 HARs ==========")

    long_df = load_long_table()
    human_model, macaque_model = load_models()
    human_class_names = get_class_names(HUMAN_ADATA_PATH)
    macaque_class_names = get_class_names(MACAQUE_ADATA_PATH)

    summary_rows = []

    for case in SELECTED_CASES:
        har_id = case["har_id"]
        group = case["group"]

        print("\n" + "=" * 90)
        print(f"Processing {har_id} | {group}")

        for seq_species in TARGET_SPECIES:
            print("-" * 60)
            print(f"Sequence species: {seq_species}")

            row_h = get_case_row(long_df, har_id, seq_species, "human_model")
            row_m = get_case_row(long_df, har_id, seq_species, "macaque_model")

            if row_h is None or row_m is None:
                print(f"[SKIP] missing row for har_id={har_id}, seq_species={seq_species}")
                continue
            
            region_str, seq = fetch_sequence(row_h)
            print("Region:", region_str)
            print("Sequence length:", len(seq))

            out_prefix = OUT_DIR / f"{group}__{sanitize_name(har_id)}"

            target_topic_h = str(row_h["top_topic"])
            target_topic_m = str(row_m["top_topic"])

            res_h = run_contribution_and_plot(
                sequence=seq,
                har_id=har_id,
                seq_species=seq_species,
                region_str=region_str,
                model=human_model,
                model_label="human_model",
                target_topic=target_topic_h,
                class_names=human_class_names,
                out_prefix=out_prefix,
            )

            res_m = run_contribution_and_plot(
                sequence=seq,
                har_id=har_id,
                seq_species=seq_species,
                region_str=region_str,
                model=macaque_model,
                model_label="macaque_model",
                target_topic=target_topic_m,
                class_names=macaque_class_names,
                out_prefix=out_prefix,
            )

            save_compare_plot(
                har_id=har_id,
                seq_species=seq_species,
                region_str=region_str,
                vec_h=res_h["profile_abs"],
                vec_m=res_m["profile_abs"],
                out_prefix=out_prefix,
            )

            summary_rows.append({
                "group": group,
                "har_id": har_id,
                "seq_species": seq_species,
                "region": region_str,
                "human_model_top_topic": target_topic_h,
                "human_model_top_score": float(row_h["top_score"]),
                "human_model_best_celltype": str(row_h["best_celltype_by_mean"]),
                "human_model_max_abs_contrib": res_h["max_abs_contrib"],
                "human_model_sum_abs_contrib": res_h["sum_abs_contrib"],
                "macaque_model_top_topic": target_topic_m,
                "macaque_model_top_score": float(row_m["top_score"]),
                "macaque_model_best_celltype": str(row_m["best_celltype_by_mean"]),
                "macaque_model_max_abs_contrib": res_m["max_abs_contrib"],
                "macaque_model_sum_abs_contrib": res_m["sum_abs_contrib"],
            })

    summary_df = pd.DataFrame(summary_rows)
    out_summary = OUT_DIR / "selected6_crossseq_contribution_summary.csv"
    summary_df.to_csv(out_summary, index=False)
    print("\nSaved:", out_summary)
    print("Done.")


if __name__ == "__main__":
    main()