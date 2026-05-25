from __future__ import annotations
import os
os.environ["KERAS_BACKEND"] = "torch"

from pathlib import Path
import keras
import pandas as pd
import numpy as np
import anndata as ad
import crested


# =========================
# paths: model / labels
# =========================
MODEL_PATH = Path("/mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/pjt/runs/out/deeptopic_macaque/final_model.keras")
ADATA_PATH = Path("/mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/pjt/runs/out/macaque_topics.h5ad")
OUT_DIR = Path("/mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/pjt/runs/out/har_3species_mscores")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# BED files
# =========================
HUMAN_BED = Path("/mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/pjt/data/active_HARs_qc_pass.bed")
CHIMP_BED = Path("/mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/pjt/data/active_HARs_pantro6_qc_pass.bed")
MACAQUE_BED = Path("/mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/pjt/data/output_rheMac10.bed")

# =========================
# genomes
# =========================
HUMAN_FA = Path("/mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/genomes/hg38/hg38.fa")
HUMAN_CHROMSIZES = Path("/mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/genomes/hg38/hg38.chrom.sizes")

CHIMP_FA = Path("/mimer/NOBACKUP/groups/naiss2025-22-612/yiquan/har/panTro6.fa")
CHIMP_CHROMSIZES = Path("/mimer/NOBACKUP/groups/naiss2025-22-612/yiquan/har/panTro6.chrom.sizes")

MACAQUE_FA = Path("/mimer/NOBACKUP/groups/naiss2025-22-612/yiquan/macaque/rheMac10.fa")
MACAQUE_CHROMSIZES = Path("/mimer/NOBACKUP/groups/naiss2025-22-612/yiquan/macaque/rheMac10.chrom.sizes")

TARGET_LEN = 500


def score_one_bed(
    bed_path: Path,
    genome_fa: Path,
    chromsizes: Path,
    model,
    class_names: list[str],
    species_label: str,
    out_csv: Path,
) -> pd.DataFrame:
    print(f"\n========== {species_label} ==========")
    print("BED:", bed_path)
    print("FA:", genome_fa)
    print("chromsizes:", chromsizes)

    # register genome for this species
    genome = crested.Genome(genome_fa, chromsizes)
    crested.register_genome(genome)

    # read BED
    har = pd.read_csv(bed_path, sep="\t", header=None, comment="#")
    if har.shape[1] < 3:
        raise ValueError(f"{bed_path} has fewer than 3 columns")

    if har.shape[1] >= 4:
        har = har.iloc[:, :4].copy()
        har.columns = ["chrom", "start", "end", "name"]
    else:
        har = har.iloc[:, :3].copy()
        har.columns = ["chrom", "start", "end"]
        har["name"] = (
            har["chrom"].astype(str)
            + ":"
            + har["start"].astype(str)
            + "-"
            + har["end"].astype(str)
        )

    har["start"] = har["start"].astype(int)
    har["end"] = har["end"].astype(int)

    # keep original id
    har["har_id"] = har["name"].astype(str)

    # resize to model input length
    mid = ((har["start"] + har["end"]) // 2).astype(int)
    har["seq_start"] = (mid - TARGET_LEN // 2).astype(int)
    har["seq_end"] = (mid + TARGET_LEN // 2).astype(int)
    har["seq_start"] = har["seq_start"].clip(lower=0)

    # fetch sequence
    def fetch_seq(row) -> str:
        return genome.fetch(row["chrom"], int(row["seq_start"]), int(row["seq_end"]))

    har["sequence"] = har.apply(fetch_seq, axis=1)
    har["seq_len"] = har["sequence"].str.len()
    har["seq_preview"] = har["sequence"].str[:20]

    print("Sequence length counts:")
    print(har["seq_len"].value_counts().head())

    har = har.loc[har["seq_len"] == TARGET_LEN].copy()
    print(f"{species_label}: kept after length filter =", len(har))

    if len(har) == 0:
        raise ValueError(f"No valid {TARGET_LEN}bp sequences kept for {species_label}")

    # predict
    seqs = har["sequence"].tolist()
    pred = crested.tl.predict(seqs, model=model)
    pred = np.asarray(pred)

    print("Prediction shape:", pred.shape)
    assert pred.shape[1] == len(class_names), (
        f"Prediction columns ({pred.shape[1]}) != class_names ({len(class_names)})"
    )

    # topic table
    topic_df = pd.DataFrame(pred, columns=class_names, index=har["har_id"])

    # summary metrics
    summary_df = pd.DataFrame(index=har["har_id"])
    summary_df["top_topic"] = topic_df.idxmax(axis=1)
    summary_df["top_score"] = pred.max(axis=1)
    summary_df["second_score"] = np.sort(pred, axis=1)[:, -2]
    summary_df["margin_top1_top2"] = summary_df["top_score"] - summary_df["second_score"]

    eps = 1e-12
    p = pred / (pred.sum(axis=1, keepdims=True) + eps)
    summary_df["entropy"] = -(p * np.log2(p + eps)).sum(axis=1)

    pred_df = pd.concat([topic_df, summary_df], axis=1)

    out = har[
        ["har_id", "chrom", "start", "end", "name", "seq_start", "seq_end", "seq_len", "seq_preview"]
    ].copy()
    out["species"] = species_label

    out = out.merge(pred_df.reset_index(), on="har_id", how="left")
    out.to_csv(out_csv, index=False)
    print("Saved:", out_csv)

    print(out[["har_id", "top_topic", "top_score", "second_score", "margin_top1_top2", "entropy"]].head())
    return out


def main():
    # load model once
    model = keras.models.load_model(str(MODEL_PATH), compile=False)
    print("Model input shape:", model.input_shape)

    adata = ad.read_h5ad(ADATA_PATH)
    class_names = list(adata.obs_names)
    print("Number of topics/classes:", len(class_names))
    print("First 10 class names:", class_names[:10])

    # score human
    human_out = score_one_bed(
        bed_path=HUMAN_BED,
        genome_fa=HUMAN_FA,
        chromsizes=HUMAN_CHROMSIZES,
        model=model,
        class_names=class_names,
        species_label="human",
        out_csv=OUT_DIR / "human_activeHAR_scores_macaqueModel.csv",
    )

    # score chimp
    chimp_out = score_one_bed(
        bed_path=CHIMP_BED,
        genome_fa=CHIMP_FA,
        chromsizes=CHIMP_CHROMSIZES,
        model=model,
        class_names=class_names,
        species_label="chimp",
        out_csv=OUT_DIR / "chimp_activeHAR_scores_macaqueModel.csv",
    )

    # score macaque
    macaque_out = score_one_bed(
        bed_path=MACAQUE_BED,
        genome_fa=MACAQUE_FA,
        chromsizes=MACAQUE_CHROMSIZES,
        model=model,
        class_names=class_names,
        species_label="macaque",
        out_csv=OUT_DIR / "macaque_activeHAR_scores_macaqueModel.csv",
    )

    # merge all
    all_out = pd.concat([human_out, chimp_out, macaque_out], axis=0, ignore_index=True)
    all_csv = OUT_DIR / "all_3species_activeHAR_scores_macaqueModel.csv"
    all_out.to_csv(all_csv, index=False)
    print("\nSaved merged file:", all_csv)


if __name__ == "__main__":
    main()