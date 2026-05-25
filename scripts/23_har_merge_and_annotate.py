from __future__ import annotations
import os
os.environ["KERAS_BACKEND"] = "torch"

from pathlib import Path
import pandas as pd
import numpy as np


# =========================
# paths
# =========================
HUMAN_MODEL_CSV = Path(
    "/mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/pjt/runs/out/har_3species_hscores/all_3species_activeHAR_scores_humanModel.csv"
)
MACAQUE_MODEL_CSV = Path(
    "/mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/pjt/runs/out/har_3species_mscores/all_3species_activeHAR_scores_macaqueModel.csv"
)

HUMAN_TOPIC_TOP_CELLTYPE = Path(
    "/mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/pjt/data/human_topic_top_celltype.tsv"
)
MACAQUE_TOPIC_TOP_CELLTYPE = Path(
    "/mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/pjt/data/macaque_topic_top_celltype.tsv"
)

HUMAN_CELLTYPE_TOPIC_MEAN = Path(
    "/mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/pjt/data/human_celltype_topic_mean.tsv"
)
MACAQUE_CELLTYPE_TOPIC_MEAN = Path(
    "/mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/pjt/data/macaque_celltype_topic_mean.tsv"
)

OUT_DIR = Path(
    "/mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/pjt/runs/out/har_integrated"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOPK = 50


# =========================
# helpers
# =========================
def natural_topic_key(x: str):
    s = str(x)
    if s.startswith("Topic"):
        try:
            return int(s.replace("Topic", ""))
        except Exception:
            return s
    return s


def find_topic_cols(df: pd.DataFrame) -> list[str]:
    topic_cols = [c for c in df.columns if str(c).startswith("Topic")]
    topic_cols = sorted(topic_cols, key=natural_topic_key)
    if len(topic_cols) == 0:
        raise ValueError("No Topic columns found in input dataframe.")
    return topic_cols


def add_summary_if_missing(df: pd.DataFrame, topic_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    pred = out[topic_cols].to_numpy(dtype=float)

    if "top_topic" not in out.columns:
        top_idx = np.argmax(pred, axis=1)
        out["top_topic"] = [topic_cols[i] for i in top_idx]

    if "top_score" not in out.columns:
        out["top_score"] = pred.max(axis=1)

    if "second_score" not in out.columns:
        out["second_score"] = np.sort(pred, axis=1)[:, -2]

    if "margin_top1_top2" not in out.columns:
        out["margin_top1_top2"] = out["top_score"] - out["second_score"]

    if "entropy" not in out.columns:
        eps = 1e-12
        p = pred / (pred.sum(axis=1, keepdims=True) + eps)
        out["entropy"] = -(p * np.log2(p + eps)).sum(axis=1)

    return out


def load_topic_top_celltype(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    print(f"\nLoaded topic_top_celltype: {path}")
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())

    required = ["topic", "top1_celltype"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"{path} missing required column: {c}")

    out_cols = ["topic", "top1_celltype"]
    if "top1_score" in df.columns:
        out_cols.append("top1_score")
    if "top2_celltype" in df.columns:
        out_cols.append("top2_celltype")
    if "top2_score" in df.columns:
        out_cols.append("top2_score")
    if "specificity_ratio_top1_over_top2" in df.columns:
        out_cols.append("specificity_ratio_top1_over_top2")

    out = df[out_cols].copy()
    out = out.rename(
        columns={
            "topic": "top_topic",
            "top1_celltype": "top_celltype",
            "top1_score": "topic_top1_celltype_mean_score",
            "top2_celltype": "topic_top2_celltype",
            "top2_score": "topic_top2_celltype_mean_score",
            "specificity_ratio_top1_over_top2": "topic_top1_vs_top2_celltype_ratio",
        }
    )
    out["top_topic"] = out["top_topic"].astype(str)
    return out


def load_celltype_topic_mean(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    print(f"\nLoaded celltype_topic_mean: {path}")
    print("Shape:", df.shape)
    print("Columns[:10]:", df.columns.tolist()[:10])

    topic_cols = [c for c in df.columns if str(c).startswith("Topic")]
    if len(topic_cols) == 0:
        raise ValueError(f"{path} has no Topic columns")

    topic_cols = sorted(topic_cols, key=natural_topic_key)

    label_col = None
    candidate_cols = ["annotation", "final_class"]
    for c in candidate_cols:
        if c in df.columns:
            label_col = c
            break

    if label_col is None:
        non_topic_cols = [c for c in df.columns if c not in topic_cols]
        if len(non_topic_cols) == 0:
            raise ValueError(f"{path} has no non-topic label column")
        label_col = non_topic_cols[0]
        print(f"Warning: using first non-topic column as label_col -> {label_col}")

    df = df[[label_col] + topic_cols].copy()
    df = df.rename(columns={label_col: "annotation"})
    df["annotation"] = df["annotation"].astype(str)

    return df


def compute_celltype_scores(
    pred_df: pd.DataFrame,
    topic_cols: list[str],
    celltype_topic_mean_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # score(celltype) = sum_topic [ HAR_topic_score * mean(celltype, topic) ]
    pred_mat = pred_df[topic_cols].to_numpy(dtype=float)

    ct_df = celltype_topic_mean_df.copy()
    ct_names = ct_df["annotation"].tolist()
    weight_mat = ct_df[topic_cols].to_numpy(dtype=float)  # shape: n_celltypes x n_topics

    # pred: n_har x n_topics
    # weight.T: n_topics x n_celltypes
    celltype_scores = pred_mat @ weight_mat.T  # shape: n_har x n_celltypes

    celltype_score_df = pd.DataFrame(
        celltype_scores,
        columns=[f"cellscore__{x}" for x in ct_names],
        index=pred_df.index,
    )

    best_idx = np.argmax(celltype_scores, axis=1)
    best_scores = celltype_scores.max(axis=1)

    tmp = np.sort(celltype_scores, axis=1)
    if tmp.shape[1] >= 2:
        second_best_scores = tmp[:, -2]
    else:
        second_best_scores = np.zeros(tmp.shape[0], dtype=float)

    summary = pd.DataFrame(index=pred_df.index)
    summary["best_celltype_by_mean"] = [ct_names[i] for i in best_idx]
    summary["best_celltype_score"] = best_scores
    summary["second_best_celltype_score"] = second_best_scores
    summary["celltype_margin_top1_top2"] = (
        summary["best_celltype_score"] - summary["second_best_celltype_score"]
    )

    return celltype_score_df, summary


def load_one_model_table(
    csv_path: Path,
    model_label: str,
    topic_top_celltype_path: Path,
    celltype_topic_mean_path: Path,
) -> pd.DataFrame:
    print(f"\n========== loading {model_label} ==========")
    print("CSV:", csv_path)

    df = pd.read_csv(csv_path)
    print("Input shape:", df.shape)

    topic_cols = find_topic_cols(df)
    print("Number of topic columns:", len(topic_cols))
    print("First 10 topic columns:", topic_cols[:10])

    df = add_summary_if_missing(df, topic_cols)

    df["har_id"] = df["har_id"].astype(str)
    df["species"] = df["species"].astype(str)
    df["model"] = model_label

    # top topic -> top celltype
    topic2cell = load_topic_top_celltype(topic_top_celltype_path)
    df = df.merge(topic2cell, on="top_topic", how="left")

    n_missing = df["top_celltype"].isna().sum()
    if n_missing > 0:
        print(f"Warning: {model_label} has {n_missing} rows with missing top_celltype")

    # use celltype_topic_mean to calculate celltype score
    ct_mean = load_celltype_topic_mean(celltype_topic_mean_path)
    celltype_score_df, celltype_summary_df = compute_celltype_scores(
        pred_df=df,
        topic_cols=topic_cols,
        celltype_topic_mean_df=ct_mean,
    )

    df = pd.concat(
        [
            df.reset_index(drop=True),
            celltype_summary_df.reset_index(drop=True),
            celltype_score_df.reset_index(drop=True),
        ],
        axis=1,
    )

    # rank and score
    df["top_score_rank_within_model"] = df["top_score"].rank(
        ascending=False, method="dense"
    )
    df["specificity_rank_within_model"] = df["margin_top1_top2"].rank(
        ascending=False, method="dense"
    )
    df["best_celltype_score_rank_within_model"] = df["best_celltype_score"].rank(
        ascending=False, method="dense"
    )
    df["combined_score"] = df["top_score"] * df["margin_top1_top2"]

    return df


def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    front_cols = [
        "har_id",
        "species",
        "model",
        "chrom",
        "start",
        "end",
        "name",
        "seq_start",
        "seq_end",
        "seq_len",
        "seq_preview",
        "top_topic",
        "top_score",
        "second_score",
        "margin_top1_top2",
        "entropy",
        "top_celltype",
        "topic_top1_celltype_mean_score",
        "topic_top2_celltype",
        "topic_top2_celltype_mean_score",
        "topic_top1_vs_top2_celltype_ratio",
        "best_celltype_by_mean",
        "best_celltype_score",
        "second_best_celltype_score",
        "celltype_margin_top1_top2",
        "top_score_rank_within_model",
        "specificity_rank_within_model",
        "best_celltype_score_rank_within_model",
        "combined_score",
    ]
    keep_front = [c for c in front_cols if c in df.columns]
    other_cols = [c for c in df.columns if c not in keep_front]
    return df[keep_front + other_cols].copy()


def make_long_table() -> pd.DataFrame:
    human_df = load_one_model_table(
        csv_path=HUMAN_MODEL_CSV,
        model_label="human_model",
        topic_top_celltype_path=HUMAN_TOPIC_TOP_CELLTYPE,
        celltype_topic_mean_path=HUMAN_CELLTYPE_TOPIC_MEAN,
    )

    macaque_df = load_one_model_table(
        csv_path=MACAQUE_MODEL_CSV,
        model_label="macaque_model",
        topic_top_celltype_path=MACAQUE_TOPIC_TOP_CELLTYPE,
        celltype_topic_mean_path=MACAQUE_CELLTYPE_TOPIC_MEAN,
    )

    long_df = pd.concat([human_df, macaque_df], axis=0, ignore_index=True)
    long_df["har_species_model"] = (
        long_df["har_id"].astype(str)
        + "|"
        + long_df["species"].astype(str)
        + "|"
        + long_df["model"].astype(str)
    )

    long_df = reorder_columns(long_df)

    out_csv = OUT_DIR / "master_har_3species_2models_long.csv"
    long_df.to_csv(out_csv, index=False)
    print("\nSaved long table:", out_csv)
    print("Long shape:", long_df.shape)

    return long_df


def make_wide_table(long_df: pd.DataFrame) -> pd.DataFrame:
    # har_id + species align
    keep_cols = [
        "har_id",
        "species",
        "chrom",
        "start",
        "end",
        "name",
        "seq_start",
        "seq_end",
        "seq_len",
        "seq_preview",
        "top_topic",
        "top_score",
        "second_score",
        "margin_top1_top2",
        "entropy",
        "top_celltype",
        "best_celltype_by_mean",
        "best_celltype_score",
        "celltype_margin_top1_top2",
        "combined_score",
    ]
    keep_cols = [c for c in keep_cols if c in long_df.columns]

    human = long_df.loc[long_df["model"] == "human_model", keep_cols].copy()
    macaque = long_df.loc[long_df["model"] == "macaque_model", keep_cols].copy()

    rename_h = {
        c: f"{c}__human_model"
        for c in human.columns
        if c not in ["har_id", "species"]
    }
    rename_m = {
        c: f"{c}__macaque_model"
        for c in macaque.columns
        if c not in ["har_id", "species"]
    }

    human = human.rename(columns=rename_h)
    macaque = macaque.rename(columns=rename_m)

    wide = human.merge(macaque, on=["har_id", "species"], how="outer")

    # wide
    if "top_score__human_model" in wide.columns and "top_score__macaque_model" in wide.columns:
        wide["delta_top_score_human_minus_macaque_model"] = (
            wide["top_score__human_model"] - wide["top_score__macaque_model"]
        )

    if (
        "best_celltype_score__human_model" in wide.columns
        and "best_celltype_score__macaque_model" in wide.columns
    ):
        wide["delta_best_celltype_score_human_minus_macaque_model"] = (
            wide["best_celltype_score__human_model"]
            - wide["best_celltype_score__macaque_model"]
        )

    out_csv = OUT_DIR / "master_har_3species_2models_wide.csv"
    wide.to_csv(out_csv, index=False)
    print("Saved wide table:", out_csv)
    print("Wide shape:", wide.shape)

    return wide


def write_top_tables(long_df: pd.DataFrame) -> None:
    for model in ["human_model", "macaque_model"]:
        sub = long_df.loc[long_df["model"] == model].copy()

        out1 = OUT_DIR / f"top{TOPK}_by_top_score__{model}.csv"
        sub.sort_values("top_score", ascending=False).head(TOPK).to_csv(out1, index=False)
        print("Saved:", out1)

        out2 = OUT_DIR / f"top{TOPK}_by_specificity__{model}.csv"
        sub.sort_values("margin_top1_top2", ascending=False).head(TOPK).to_csv(out2, index=False)
        print("Saved:", out2)

        out3 = OUT_DIR / f"top{TOPK}_by_best_celltype_score__{model}.csv"
        sub.sort_values("best_celltype_score", ascending=False).head(TOPK).to_csv(out3, index=False)
        print("Saved:", out3)

        out4 = OUT_DIR / f"top{TOPK}_by_combined_score__{model}.csv"
        sub.sort_values("combined_score", ascending=False).head(TOPK).to_csv(out4, index=False)
        print("Saved:", out4)


def write_celltype_summary(long_df: pd.DataFrame) -> None:
    if "top_celltype" in long_df.columns:
        s1 = (
            long_df.groupby(["model", "species", "top_celltype"], dropna=False)
            .size()
            .reset_index(name="n")
            .sort_values(["model", "species", "n"], ascending=[True, True, False])
        )
        out1 = OUT_DIR / "summary_counts_by_top_celltype.csv"
        s1.to_csv(out1, index=False)
        print("Saved:", out1)

    if "best_celltype_by_mean" in long_df.columns:
        s2 = (
            long_df.groupby(["model", "species", "best_celltype_by_mean"], dropna=False)
            .size()
            .reset_index(name="n")
            .sort_values(["model", "species", "n"], ascending=[True, True, False])
        )
        out2 = OUT_DIR / "summary_counts_by_best_celltype_by_mean.csv"
        s2.to_csv(out2, index=False)
        print("Saved:", out2)



def main():
    print("========== Step: merge and annotate HAR scores ==========")

    long_df = make_long_table()
    wide_df = make_wide_table(long_df)

    write_top_tables(long_df)
    write_celltype_summary(long_df)

    print("\nDone.")
    print("Main outputs:")
    print(OUT_DIR / "master_har_3species_2models_long.csv")
    print(OUT_DIR / "master_har_3species_2models_wide.csv")


if __name__ == "__main__":
    main()