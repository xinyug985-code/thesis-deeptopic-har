from __future__ import annotations
import os
os.environ["KERAS_BACKEND"] = "torch"
import keras

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import anndata as ad
import crested


# ============================================================
# Paths
# ============================================================
BASE_DIR = Path("/mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/pjt")
OUT_BASE = BASE_DIR / "runs" / "out" / "har_integrated"
LONG_CSV = OUT_BASE / "master_har_3species_2models_long.csv"

HUMAN_MODEL_PATH = BASE_DIR / "runs" / "out" / "deeptopic_human" / "final_model.keras"
MACAQUE_MODEL_PATH = BASE_DIR / "runs" / "out" / "deeptopic_macaque" / "final_model.keras"

HUMAN_ADATA_PATH = BASE_DIR / "runs" / "out" / "human_topics.h5ad"
MACAQUE_ADATA_PATH = BASE_DIR / "runs" / "out" / "macaque_topics.h5ad"

HUMAN_FA = Path("/mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/genomes/hg38/hg38.fa")
CHIMP_FA = Path("/mimer/NOBACKUP/groups/naiss2025-22-612/yiquan/har/panTro6.fa")
MACAQUE_FA = Path("/mimer/NOBACKUP/groups/naiss2025-22-612/yiquan/macaque/rheMac10.fa")

OUT_DIR = OUT_BASE / "case_contribution"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CASES = [
    # human-biased
    {"har_id": "HARsv2_0513", "species": "macaque", "group": "human_biased"},
    # conserved
    {"har_id": "ZOOHAR.307", "species": "human", "group": "conserved"},
    # macaque-biased
    {"har_id": "HARsv2_0920", "species": "human", "group": "macaque_biased"},
]

# integrated_grad / expected_integrated_grad / mutagenesis / saliency_map 
METHOD = "integrated_grad"
ZOOM_N_BASES = 500


# ============================================================
# Helpers
# ============================================================
def sanitize_name(x: str) -> str:
    x = str(x)
    for ch in [" ", "/", "\\", "|", ":", ";", ",", "(", ")", "[", "]"]:
        x = x.replace(ch, "_")
    return x


def infer_fasta_for_species(species: str) -> Path:
    species = str(species).lower()
    if species == "human":
        return HUMAN_FA
    if species == "chimp":
        return CHIMP_FA
    if species == "macaque":
        return MACAQUE_FA
    raise ValueError(f"Unknown species: {species}")


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


def load_models():
    print("Loading models...")
    human_model = keras.models.load_model(str(HUMAN_MODEL_PATH), compile=False)
    macaque_model = keras.models.load_model(str(MACAQUE_MODEL_PATH), compile=False)
    print("Human model loaded:", HUMAN_MODEL_PATH)
    print("Macaque model loaded:", MACAQUE_MODEL_PATH)
    return human_model, macaque_model

def build_genome_for_species(species: str):
    species = str(species).lower()
    if species == "human":
        return crested.Genome(HUMAN_FA)
    if species == "chimp":
        return crested.Genome(CHIMP_FA)
    if species == "macaque":
        return crested.Genome(MACAQUE_FA)
    raise ValueError(f"Unknown species: {species}")


def load_long_table() -> pd.DataFrame:
    df = pd.read_csv(LONG_CSV)
    print("Loaded long table:", LONG_CSV)
    print("Shape:", df.shape)
    return df


def get_case_rows(long_df: pd.DataFrame, har_id: str, species: str) -> pd.DataFrame:
    sub = long_df.loc[
        (long_df["har_id"].astype(str) == str(har_id)) &
        (long_df["species"].astype(str) == str(species))
    ].copy()

    if sub.empty:
        raise ValueError(f"No rows found for har_id={har_id}, species={species}")

    return sub.sort_values("model").copy()


def fetch_sequence_from_case_row(row: pd.Series) -> tuple[str, str]:
    chrom = str(row["chrom"])
    seq_start = int(row["seq_start"])
    seq_end = int(row["seq_end"])
    region_str = f"{chrom}:{seq_start}-{seq_end}"

    genome = build_genome_for_species(str(row["species"]))
    seq = genome.fetch(chrom, seq_start, seq_end).upper()

    if len(seq) != (seq_end - seq_start):
        warnings.warn(
            f"Fetched sequence length {len(seq)} != expected {seq_end - seq_start} "
            f"for {region_str}"
        )
    return region_str, seq


def summarize_scores_to_vector(scores: np.ndarray) -> np.ndarray:
    arr = np.asarray(scores)

    # (1, 1, L, 4)
    if arr.ndim == 4:
        arr = arr[0, 0, :, :]
    elif arr.ndim == 3:
        # (1, L, 4) or (C, L, 4)
        if arr.shape[0] == 1:
            arr = arr[0, :, :]
        else:
            arr = arr.mean(axis=0)
    else:
        raise ValueError(f"Unexpected score shape: {arr.shape}")

    # summary
    vec = np.sum(np.abs(arr), axis=1)
    return vec


def save_simple_profile_plot(
    vec_h: np.ndarray,
    vec_m: np.ndarray,
    region_label: str,
    out_base: Path,
) -> None:
    fig = plt.figure(figsize=(12, 3.5))
    ax = fig.add_subplot(111)
    ax.plot(vec_h, label="human_model")
    ax.plot(vec_m, label="macaque_model")
    ax.set_title(f"Contribution profile comparison\n{region_label}")
    ax.set_xlabel("Position")
    ax.set_ylabel("Absolute contribution")
    ax.legend(frameon=False)
    fig.savefig(str(out_base) + "_profile_compare.png", dpi=300, bbox_inches="tight")
    fig.savefig(str(out_base) + "_profile_compare.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved:", str(out_base) + "_profile_compare.png")
    print("Saved:", str(out_base) + "_profile_compare.pdf")


def compute_and_plot_one_model(
    sequence: str,
    region_str: str,
    model,
    class_names: list[str],
    target_topic: str,
    model_label: str,
    out_base: Path,
):
    target_idx = get_target_idx(class_names, target_topic)

    print(f"\n[{model_label}] target_topic = {target_topic}, target_idx = {target_idx}")
    scores, seqs_one_hot = crested.tl.contribution_scores(
        sequence,
        target_idx=target_idx,
        model=model,
        method=METHOD,
        batch_size=128,
        verbose=True,
    )

    arr = np.asarray(scores)

    # (1, 1, L, 4) or (1, L, 4)
    if arr.ndim == 4:
        arr_plot = arr[0, 0, :, :]
    elif arr.ndim == 3:
        if arr.shape[0] == 1:
            arr_plot = arr[0, :, :]
        else:
            arr_plot = arr.mean(axis=0)
    else:
        raise ValueError(f"Unexpected score shape: {arr.shape}")

    if arr_plot.shape[1] != 4:
        raise ValueError(f"Expected last dimension = 4, got {arr_plot.shape}")

    # position-level profile
    profile_abs = np.sum(np.abs(arr_plot), axis=1)
    profile_signed = np.sum(arr_plot, axis=1)

    # -------------------------
    # position contribution profile
    # -------------------------
    fig = plt.figure(figsize=(12, 3.5))
    ax = fig.add_subplot(111)
    ax.plot(profile_abs, label="abs contribution")
    ax.plot(profile_signed, label="signed contribution", alpha=0.8)
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_title(f"{model_label} contribution profile\n{target_topic} | {region_str}")
    ax.set_xlabel("Position")
    ax.set_ylabel("Contribution")
    ax.legend(frameon=False)
    fig.savefig(str(out_base) + f"__{model_label}__profile.png", dpi=300, bbox_inches="tight")
    fig.savefig(str(out_base) + f"__{model_label}__profile.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved:", str(out_base) + f"__{model_label}__profile.png")
    print("Saved:", str(out_base) + f"__{model_label}__profile.pdf")

    # -------------------------
    # A/C/G/T contribution heatmap
    # -------------------------
    fig = plt.figure(figsize=(14, 3.8))
    ax = fig.add_subplot(111)

    im = ax.imshow(
        arr_plot.T,
        aspect="auto",
        interpolation="nearest",
        origin="lower",
    )
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(["A", "C", "G", "T"])
    ax.set_xlabel("Position")
    ax.set_title(f"{model_label} base-wise contribution heatmap\n{target_topic} | {region_str}")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Contribution score")

    fig.savefig(str(out_base) + f"__{model_label}__base_heatmap.png", dpi=300, bbox_inches="tight")
    fig.savefig(str(out_base) + f"__{model_label}__base_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved:", str(out_base) + f"__{model_label}__base_heatmap.png")
    print("Saved:", str(out_base) + f"__{model_label}__base_heatmap.pdf")

    raw_csv = str(out_base) + f"__{model_label}__base_contrib_matrix.csv"
    pd.DataFrame(arr_plot, columns=["A", "C", "G", "T"]).to_csv(raw_csv, index=False)
    print("Saved:", raw_csv)

    return {
        "target_topic": target_topic,
        "target_idx": target_idx,
        "scores": scores,
        "seqs_one_hot": seqs_one_hot,
        "profile": profile_abs,
        "profile_signed": profile_signed,
        "base_matrix": arr_plot,
    }


# ============================================================
# Main
# ============================================================
def main():
    print("========== Case contribution score analysis ==========")
    print("Method:", METHOD)

    long_df = load_long_table()
    human_model, macaque_model = load_models()

    human_class_names = get_class_names(HUMAN_ADATA_PATH)
    macaque_class_names = get_class_names(MACAQUE_ADATA_PATH)

    summary_rows = []

    for case in CASES:
        har_id = case["har_id"]
        species = case["species"]
        group = case["group"]

        print("\n" + "=" * 80)
        print(f"Processing: {har_id} | {species} | {group}")

        case_rows = get_case_rows(long_df, har_id=har_id, species=species)

        region_str, seq = fetch_sequence_from_case_row(case_rows.iloc[0])
        print("Region:", region_str)
        print("Sequence length:", len(seq))

        out_base = OUT_DIR / f"{group}__{sanitize_name(har_id)}__{sanitize_name(species)}"

        row_h = case_rows.loc[case_rows["model"] == "human_model"].iloc[0]
        row_m = case_rows.loc[case_rows["model"] == "macaque_model"].iloc[0]

        target_topic_h = str(row_h["top_topic"])
        target_topic_m = str(row_m["top_topic"])

        # human model contribution
        res_h = compute_and_plot_one_model(
            sequence=seq,
            region_str=region_str,
            model=human_model,
            class_names=human_class_names,
            target_topic=target_topic_h,
            model_label="human_model",
            out_base=out_base,
        )

        # macaque model contribution
        res_m = compute_and_plot_one_model(
            sequence=seq,
            region_str=region_str,
            model=macaque_model,
            class_names=macaque_class_names,
            target_topic=target_topic_m,
            model_label="macaque_model",
            out_base=out_base,
        )

        save_simple_profile_plot(
            vec_h=res_h["profile"],
            vec_m=res_m["profile"],
            region_label=f"{har_id} | {species} | {region_str}",
            out_base=out_base,
        )

        # save profile 
        profile_df = pd.DataFrame({
            "position": np.arange(len(res_h["profile"])),
            "human_model_abs_contrib": res_h["profile"],
            "macaque_model_abs_contrib": res_m["profile"],
        })
        profile_csv = str(out_base) + "__profile_compare.csv"
        profile_df.to_csv(profile_csv, index=False)
        print("Saved:", profile_csv)

        # summary
        summary_rows.append({
            "group": group,
            "har_id": har_id,
            "species": species,
            "region": region_str,
            "seq_len": len(seq),
            "human_model_top_topic": target_topic_h,
            "human_model_target_idx": res_h["target_idx"],
            "human_model_top_score": float(row_h["top_score"]),
            "human_model_best_celltype": str(row_h.get("best_celltype_by_mean", "")),
            "macaque_model_top_topic": target_topic_m,
            "macaque_model_target_idx": res_m["target_idx"],
            "macaque_model_top_score": float(row_m["top_score"]),
            "macaque_model_best_celltype": str(row_m.get("best_celltype_by_mean", "")),
            "max_abs_contrib_human_model": float(np.max(res_h["profile"])),
            "max_abs_contrib_macaque_model": float(np.max(res_m["profile"])),
            "sum_abs_contrib_human_model": float(np.sum(res_h["profile"])),
            "sum_abs_contrib_macaque_model": float(np.sum(res_m["profile"])),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = OUT_DIR / "case_contribution_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print("\nSaved summary:", summary_csv)
    print("\nDone.")


if __name__ == "__main__":
    main()