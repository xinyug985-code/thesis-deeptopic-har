from __future__ import annotations
import os
os.environ["KERAS_BACKEND"] = "torch"

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import keras
import anndata as ad
import crested


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

OUT_DIR = OUT_BASE / "lost9_har_contribution"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 这9个 liftOver failed HAR
LOST_HARS = [
    "HARsv2_0021",
    "HARsv2_0327",
    "HARsv2_1082",
    "HARsv2_1136",
    "HARsv2_1309",
    "HARsv2_1586",
    "HARsv2_1690",
    "HARsv2_1943",
    "HARsv2_2153",
]

METHOD = "integrated_grad"

SAVE_BASE_HEATMAP = True


# ============================================================
# helpers
# ============================================================
def sanitize_name(x: str) -> str:
    x = str(x)
    for ch in [" ", "/", "\\", "|", ":", ";", ",", "(", ")", "[", "]"]:
        x = x.replace(ch, "_")
    return x


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
            f"Target topic {target_topic} not found. First 10: {class_names[:10]}"
        )
    return class_names.index(target_topic)


def load_long_table() -> pd.DataFrame:
    df = pd.read_csv(LONG_CSV)
    print("Loaded long table:", LONG_CSV)
    print("Shape:", df.shape)
    return df


def get_case_rows_from_human_input(long_df: pd.DataFrame, har_id: str) -> pd.DataFrame:
    """
    - human_model on human sequence
    - macaque_model on human sequence
    """
    sub = long_df.loc[
        (long_df["har_id"].astype(str) == str(har_id)) &
        (long_df["species"].astype(str) == "human")
    ].copy()

    if sub.empty:
        raise ValueError(f"No rows found for har_id={har_id} with species=human")

    models_present = set(sub["model"].astype(str))
    if models_present != {"human_model", "macaque_model"}:
        raise ValueError(
            f"{har_id} does not contain both human_model and macaque_model rows. "
            f"Found: {models_present}"
        )

    return sub.sort_values("model").copy()


def fetch_human_sequence(row: pd.Series) -> tuple[str, str]:
    chrom = str(row["chrom"])
    seq_start = int(row["seq_start"])
    seq_end = int(row["seq_end"])
    region_str = f"{chrom}:{seq_start}-{seq_end}"

    genome = crested.Genome(HUMAN_FA)
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

    raise ValueError(f"Unexpected score shape: {arr.shape}")


def compute_profile(arr_plot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    profile_abs = np.sum(np.abs(arr_plot), axis=1)
    profile_signed = np.sum(arr_plot, axis=1)
    return profile_abs, profile_signed


def save_single_model_profile(
    profile_abs: np.ndarray,
    profile_signed: np.ndarray,
    model_label: str,
    target_topic: str,
    region_str: str,
    out_base: Path,
) -> None:
    fig = plt.figure(figsize=(12, 3.5))
    ax = fig.add_subplot(111)
    ax.plot(profile_abs, label="abs contribution")
    ax.plot(profile_signed, label="signed contribution")
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_title(f"{model_label} contribution profile\n{target_topic} | {region_str}")
    ax.set_xlabel("Position")
    ax.set_ylabel("Contribution")
    ax.legend(frameon=False)

    fig.savefig(str(out_base) + f"__{model_label}__profile.png", dpi=300, bbox_inches="tight")
    fig.savefig(str(out_base) + f"__{model_label}__profile.pdf", bbox_inches="tight")
    plt.close(fig)


def save_base_heatmap(
    arr_plot: np.ndarray,
    model_label: str,
    target_topic: str,
    region_str: str,
    out_base: Path,
) -> None:
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


def save_compare_profile(
    vec_h: np.ndarray,
    vec_m: np.ndarray,
    har_id: str,
    region_str: str,
    out_base: Path,
) -> None:
    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(111)
    ax.plot(vec_h, label="human_model")
    ax.plot(vec_m, label="macaque_model")
    ax.set_title(f"Contribution profile comparison\n{har_id} | human | {region_str}")
    ax.set_xlabel("Position")
    ax.set_ylabel("Absolute contribution")
    ax.legend(frameon=False)

    fig.savefig(str(out_base) + "__profile_compare.png", dpi=300, bbox_inches="tight")
    fig.savefig(str(out_base) + "__profile_compare.pdf", bbox_inches="tight")
    plt.close(fig)


def compute_one_model(
    sequence: str,
    model,
    class_names: list[str],
    target_topic: str,
):
    target_idx = get_target_idx(class_names, target_topic)

    scores, _ = crested.tl.contribution_scores(
        sequence,
        target_idx=target_idx,
        model=model,
        method=METHOD,
        batch_size=128,
        verbose=True,
    )

    arr_plot = extract_plot_matrix(scores)
    profile_abs, profile_signed = compute_profile(arr_plot)

    return {
        "target_topic": target_topic,
        "target_idx": target_idx,
        "scores": scores,
        "base_matrix": arr_plot,
        "profile_abs": profile_abs,
        "profile_signed": profile_signed,
        "max_abs": float(np.max(profile_abs)),
        "sum_abs": float(np.sum(profile_abs)),
    }


# ============================================================
# main
# ============================================================
def main():
    print("========== Batch contribution for lost 9 HARs ==========")
    print("Method:", METHOD)

    long_df = load_long_table()
    human_model, macaque_model = load_models()
    human_class_names = get_class_names(HUMAN_ADATA_PATH)
    macaque_class_names = get_class_names(MACAQUE_ADATA_PATH)

    summary_rows = []

    for har_id in LOST_HARS:
        print("\n" + "=" * 80)
        print("Processing:", har_id)

        case_rows = get_case_rows_from_human_input(long_df, har_id)
        row_h = case_rows.loc[case_rows["model"] == "human_model"].iloc[0]
        row_m = case_rows.loc[case_rows["model"] == "macaque_model"].iloc[0]

        region_str, seq = fetch_human_sequence(row_h)
        print("Region:", region_str)
        print("Sequence length:", len(seq))

        out_base = OUT_DIR / sanitize_name(har_id)

        target_topic_h = str(row_h["top_topic"])
        target_topic_m = str(row_m["top_topic"])

        print(f"[human_model] target_topic={target_topic_h}")
        res_h = compute_one_model(
            sequence=seq,
            model=human_model,
            class_names=human_class_names,
            target_topic=target_topic_h,
        )

        print(f"[macaque_model] target_topic={target_topic_m}")
        res_m = compute_one_model(
            sequence=seq,
            model=macaque_model,
            class_names=macaque_class_names,
            target_topic=target_topic_m,
        )

        save_single_model_profile(
            profile_abs=res_h["profile_abs"],
            profile_signed=res_h["profile_signed"],
            model_label="human_model",
            target_topic=target_topic_h,
            region_str=region_str,
            out_base=out_base,
        )
        save_single_model_profile(
            profile_abs=res_m["profile_abs"],
            profile_signed=res_m["profile_signed"],
            model_label="macaque_model",
            target_topic=target_topic_m,
            region_str=region_str,
            out_base=out_base,
        )

        if SAVE_BASE_HEATMAP:
            save_base_heatmap(
                arr_plot=res_h["base_matrix"],
                model_label="human_model",
                target_topic=target_topic_h,
                region_str=region_str,
                out_base=out_base,
            )
            save_base_heatmap(
                arr_plot=res_m["base_matrix"],
                model_label="macaque_model",
                target_topic=target_topic_m,
                region_str=region_str,
                out_base=out_base,
            )

        save_compare_profile(
            vec_h=res_h["profile_abs"],
            vec_m=res_m["profile_abs"],
            har_id=har_id,
            region_str=region_str,
            out_base=out_base,
        )

        # profile csv
        profile_df = pd.DataFrame({
            "position": np.arange(len(res_h["profile_abs"])),
            "human_model_abs_contrib": res_h["profile_abs"],
            "human_model_signed_contrib": res_h["profile_signed"],
            "macaque_model_abs_contrib": res_m["profile_abs"],
            "macaque_model_signed_contrib": res_m["profile_signed"],
        })
        profile_df.to_csv(str(out_base) + "__profile_compare.csv", index=False)

        summary_rows.append({
            "har_id": har_id,
            "region": region_str,
            "human_model_top_topic": target_topic_h,
            "human_model_top_score": float(row_h["top_score"]),
            "human_model_best_celltype": str(row_h["best_celltype_by_mean"]),
            "human_model_target_idx": int(res_h["target_idx"]),
            "human_model_max_abs_contrib": res_h["max_abs"],
            "human_model_sum_abs_contrib": res_h["sum_abs"],
            "macaque_model_top_topic": target_topic_m,
            "macaque_model_top_score": float(row_m["top_score"]),
            "macaque_model_best_celltype": str(row_m["best_celltype_by_mean"]),
            "macaque_model_target_idx": int(res_m["target_idx"]),
            "macaque_model_max_abs_contrib": res_m["max_abs"],
            "macaque_model_sum_abs_contrib": res_m["sum_abs"],
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = OUT_DIR / "lost9_har_contribution_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    print("\nSaved summary:", summary_csv)
    print("Done.")


if __name__ == "__main__":
    main()