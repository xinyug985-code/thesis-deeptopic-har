from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

from pycisTopic.topic_binarization import binarize_topics
from pycisTopic.topic_qc import compute_topic_metrics, topic_annotation


# =====================
# Config
# =====================
BASE = Path("/cfs/klemming/projects/supr/snic2022-23-547/xinyu")
OUT = BASE / "runs" / "out"

IN_COBJ = OUT / "1_cobj_macaque.withmeta.pkl"
IN_MODEL = OUT / "2_cobj_tmp_mallet_macaque_k40-100" / "Topic100.pkl"

OUT_CELL_BIN_PKL = OUT / "3_cell_topic_binarized_macaque_li.pkl"
OUT_TOPIC_ANNOT_TSV = OUT / "3_topic_annotation_macaque_pycistopic.tsv"
OUT_TOPIC_QC_TSV = OUT / "3_topic_qc_metrics_macaque.tsv"

ANNOT_VAR = "annotation"

METHOD = "li"
GENERAL_TOPIC_THR = 0.2


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    if not IN_COBJ.exists():
        raise FileNotFoundError(f"Missing cisTopic object: {IN_COBJ}")
    if not IN_MODEL.exists():
        raise FileNotFoundError(f"Missing model: {IN_MODEL}")

    print("[load] cistopic_obj:", IN_COBJ)
    cistopic_obj = load_pickle(IN_COBJ)

    print("[load] model:", IN_MODEL)
    model = load_pickle(IN_MODEL)

    cistopic_obj.selected_model = model
    print(f"[model] selected_model.n_topic = {model.n_topic}")

    print("[metadata columns]")
    print(list(cistopic_obj.cell_data.columns))

    if ANNOT_VAR not in cistopic_obj.cell_data.columns:
        raise ValueError(
            f"ANNOT_VAR='{ANNOT_VAR}' not found in cistopic_obj.cell_data.columns. "
            f"Available columns: {list(cistopic_obj.cell_data.columns)}"
        )

    # ============================================================
    # 1) strictly tutorial: binarize cell-topic distribution
    # ============================================================
    print(f"[cell-topic binarization] target='cell', method='{METHOD}'")
    binarized_cell_topic = binarize_topics(
        cistopic_obj,
        target="cell",
        method=METHOD,
        plot=True,
        num_columns=5,
        nbins=100,
    )

    with open(OUT_CELL_BIN_PKL, "wb") as f:
        pickle.dump(binarized_cell_topic, f)
    print("[save] binarized_cell_topic ->", OUT_CELL_BIN_PKL)

    # ============================================================
    # 2) tutorial: topic QC metrics
    # ============================================================
    print("[topic QC] compute_topic_metrics")
    topic_qc_metrics = compute_topic_metrics(cistopic_obj)

    topic_qc_metrics.to_csv(OUT_TOPIC_QC_TSV, sep="\t")
    print("[save] topic QC metrics ->", OUT_TOPIC_QC_TSV)

    # ============================================================
    # 3) strictly tutorial: automatic topic annotation by cell type
    # ============================================================
    print(f"[topic annotation] annot_var='{ANNOT_VAR}'")
    topic_annot = topic_annotation(
        cistopic_obj,
        annot_var=ANNOT_VAR,
        binarized_cell_topic=binarized_cell_topic,
        general_topic_thr=GENERAL_TOPIC_THR,
    )

    topic_annot = topic_annot.copy()
    topic_annot.index.name = "topic"

    topic_annot.to_csv(OUT_TOPIC_ANNOT_TSV, sep="\t")
    print("[save] topic annotation ->", OUT_TOPIC_ANNOT_TSV)

    print("\n[preview]")
    print(topic_annot.head(20))

    print("\n[done]")


if __name__ == "__main__":
    main()