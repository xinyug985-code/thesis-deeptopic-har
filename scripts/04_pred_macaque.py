import os
from pathlib import Path

os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import pandas as pd
import keras
import anndata as ad
import crested

BASE = Path("/mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/pjt")
OUTD = BASE / "runs" / "out"

H5AD = OUTD / "macaque_topics.h5ad"
GENOME_FA = "/mimer/NOBACKUP/groups/naiss2025-22-612/yiquan/macaque/rheMac10.fa"
MODEL_PATH = OUTD / "deeptopic_macaque" / "final_model.keras"

SAVE_DIR = OUTD / "deeptopic_macaque" / "prediction_inspection"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV_DETAIL = SAVE_DIR / "test_predictions_detailed.csv"
OUT_CSV_TOPIC_SUMMARY = SAVE_DIR / "topic_summary.csv"
OUT_CSV_CONFUSION = SAVE_DIR / "topic_confusion_pairs.csv"
OUT_H5AD = SAVE_DIR / "macaque_topics_with_predictions.h5ad"

BATCH_SIZE = 8
TOPK = 3


def main():
    # -----------------------------
    # load data
    # -----------------------------
    adata = ad.read_h5ad(str(H5AD))

    genome = crested.Genome(GENOME_FA)
    crested.register_genome(genome)

    print("Genome sanity check:", genome.fetch("chr1", 10_000_000, 10_000_010))
    print("adata shape:", adata.shape)

    if "split" not in adata.var.columns:
        raise ValueError("adata.var does not contain 'split' column")

    test_mask = (adata.var["split"] == "test")
    n_test = int(test_mask.sum())
    print("N test regions:", n_test)

    # -----------------------------
    # load model and predict
    # -----------------------------
    model = keras.models.load_model(str(MODEL_PATH), compile=False)

    predictions = crested.tl.predict(adata, model, batch_size=BATCH_SIZE)
    predictions = np.asarray(predictions)

    print("predictions shape:", predictions.shape)  # expected: (n_regions, n_topics)

    # save predictions into adata.layers
    adata.layers["pred_macaque"] = predictions.T  # (n_obs, n_vars)

    # -----------------------------
    # extract test set
    # -----------------------------
    y_true = adata.X[:, test_mask].T
    y_pred = predictions[test_mask.values, :]

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    test_regions = np.array(adata.var_names[test_mask])
    topic_names = np.array(adata.obs_names)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape}, y_pred {y_pred.shape}")

    print("y_true shape:", y_true.shape)
    print("y_pred shape:", y_pred.shape)

    # -----------------------------
    # get label indices
    # -----------------------------
    true_idx = y_true.argmax(axis=1)
    pred_idx = y_pred.argmax(axis=1)

    true_topic = topic_names[true_idx]
    pred_topic = topic_names[pred_idx]

    pred_top1_score = y_pred[np.arange(len(pred_idx)), pred_idx]
    true_topic_score = y_pred[np.arange(len(true_idx)), true_idx]

    correct_top1 = (true_idx == pred_idx)

    # -----------------------------
    # top-k predictions
    # -----------------------------
    topk_idx = np.argsort(-y_pred, axis=1)[:, :TOPK]
    topk_scores = np.take_along_axis(y_pred, topk_idx, axis=1)

    topk_topics = topic_names[topk_idx]

    # whether true label is inside top-k
    correct_topk = np.array([
        true_idx[i] in topk_idx[i]
        for i in range(len(true_idx))
    ])

    # -----------------------------
    # detailed region-level table
    # -----------------------------
    detail_dict = {
        "region": test_regions,
        "true_topic_idx": true_idx,
        "true_topic": true_topic,
        "pred_topic_idx": pred_idx,
        "pred_topic": pred_topic,
        "pred_top1_score": pred_top1_score,
        "true_topic_score": true_topic_score,
        "correct_top1": correct_top1,
        f"correct_top{TOPK}": correct_topk,
    }

    for k in range(TOPK):
        detail_dict[f"top{k+1}_topic_idx"] = topk_idx[:, k]
        detail_dict[f"top{k+1}_topic"] = topk_topics[:, k]
        detail_dict[f"top{k+1}_score"] = topk_scores[:, k]

    df_detail = pd.DataFrame(detail_dict)

    print("\nDetailed predictions preview:")
    print(df_detail.head(10))

    top1_acc = df_detail["correct_top1"].mean()
    topk_acc = df_detail[f"correct_top{TOPK}"].mean()

    print(f"\nTop1 accuracy from table: {top1_acc:.4f}")
    print(f"Top{TOPK} accuracy from table: {topk_acc:.4f}")

    # -----------------------------
    # topic-level summary
    # -----------------------------
    topic_rows = []
    for topic_i, topic_name in enumerate(topic_names):
        mask_i = (true_idx == topic_i)
        n_i = int(mask_i.sum())

        if n_i == 0:
            topic_rows.append({
                "topic_idx": topic_i,
                "topic": topic_name,
                "n_regions": 0,
                "top1_accuracy": np.nan,
                f"top{TOPK}_accuracy": np.nan,
                "mean_true_topic_score": np.nan,
                "mean_pred_top1_score": np.nan,
            })
            continue

        topic_rows.append({
            "topic_idx": topic_i,
            "topic": topic_name,
            "n_regions": n_i,
            "top1_accuracy": float(correct_top1[mask_i].mean()),
            f"top{TOPK}_accuracy": float(correct_topk[mask_i].mean()),
            "mean_true_topic_score": float(true_topic_score[mask_i].mean()),
            "mean_pred_top1_score": float(pred_top1_score[mask_i].mean()),
        })

    df_topic_summary = pd.DataFrame(topic_rows)
    df_topic_summary = df_topic_summary.sort_values(
        by=["top1_accuracy", "n_regions"],
        ascending=[False, False]
    )

    print("\nTopic summary preview:")
    print(df_topic_summary.head(10))

    # -----------------------------
    # topic confusion pairs
    # -----------------------------
    wrong_mask = ~correct_top1
    df_wrong = pd.DataFrame({
        "true_topic": true_topic[wrong_mask],
        "pred_topic": pred_topic[wrong_mask],
    })

    if len(df_wrong) > 0:
        df_confusion = (
            df_wrong
            .groupby(["true_topic", "pred_topic"])
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
    else:
        df_confusion = pd.DataFrame(columns=["true_topic", "pred_topic", "count"])

    print("\nMost frequent confusion pairs:")
    print(df_confusion.head(20))

    # -----------------------------
    # save
    # -----------------------------
    df_detail.to_csv(OUT_CSV_DETAIL, index=False)
    df_topic_summary.to_csv(OUT_CSV_TOPIC_SUMMARY, index=False)
    df_confusion.to_csv(OUT_CSV_CONFUSION, index=False)
    adata.write_h5ad(str(OUT_H5AD))

    print("\nSaved files:")
    print(" -", OUT_CSV_DETAIL)
    print(" -", OUT_CSV_TOPIC_SUMMARY)
    print(" -", OUT_CSV_CONFUSION)
    print(" -", OUT_H5AD)


if __name__ == "__main__":
    main()