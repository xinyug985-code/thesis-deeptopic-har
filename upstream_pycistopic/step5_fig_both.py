from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

import numpy as np
import pandas as pd

from pycisTopic.clust_vis import (
    find_clusters,
    run_umap,
    run_tsne,
    plot_metadata,
    plot_topic,
    cell_topic_heatmap,
)


BASE = Path("/cfs/klemming/projects/supr/snic2022-23-547/xinyu")
OUT = BASE / "runs" / "out"

DATASETS = {
    "human": {
        "cobj": OUT / "1_cobj_human.withmeta.pkl",
        "model": OUT / "2_cobj_tmp_mallet_human_k40-100" / "Topic100.pkl",
        "label_col": "final_class",
    },
    "macaque": {
        "cobj": OUT / "1_cobj_macaque.withmeta.pkl",
        "model": OUT / "2_cobj_tmp_mallet_macaque_k40-100" / "Topic100.pkl",
        "label_col": "annotation",
    },
}

FIGDIR = OUT / "clust_vis"
FIGDIR.mkdir(parents=True, exist_ok=True)

TABDIR = OUT / "clust_vis_tables"
TABDIR.mkdir(parents=True, exist_ok=True)


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def save_current(fig_path: Path, dpi=180):
    import matplotlib.pyplot as plt
    fig_path = fig_path.with_suffix(".png")
    plt.savefig(fig_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print("[save fig]", fig_path)

def plot_celltype_topic_mean_heatmap(tag: str, cobj, model, label_col: str):
    import matplotlib.pyplot as plt

    cell_topic_df = get_cell_topic_df(cobj, model)

    meta = cobj.cell_data[[label_col]].copy()
    meta = meta.loc[meta[label_col].notna()].copy()

    merged = meta.join(cell_topic_df, how="inner")
    topic_cols = [c for c in merged.columns if c.startswith("Topic")]

    mat = merged.groupby(label_col)[topic_cols].mean()

    # row-wise z-score
    mat_z = mat.copy()
    mat_z = mat_z.sub(mat_z.mean(axis=1), axis=0)
    mat_z = mat_z.div(mat_z.std(axis=1).replace(0, np.nan), axis=0)
    mat_z = mat_z.fillna(0)

    fig_h = max(6, 0.35 * mat_z.shape[0])
    fig_w = max(18, 0.18 * mat_z.shape[1])

    plt.figure(figsize=(fig_w, fig_h))
    im = plt.imshow(mat_z.values, aspect="auto", cmap="turbo", vmin=-2, vmax=2)

    plt.yticks(
        ticks=np.arange(mat_z.shape[0]),
        labels=mat_z.index,
        fontsize=7,
    )

    plt.xticks(
        ticks=np.arange(mat_z.shape[1]),
        labels=mat_z.columns,
        rotation=90,
        fontsize=5,
    )

    plt.colorbar(im, fraction=0.02, pad=0.01, label="row z-score mean topic usage")
    plt.title(f"{tag}: cell type × topic mean activity")
    plt.xlabel("Topic")
    plt.ylabel("Cell type")

    out = FIGDIR / f"{tag}_celltype_topic_mean_heatmap.png"
    plt.savefig(out, dpi=220, bbox_inches="tight")
    plt.close()

    print("[save heatmap]", out)

def pick_existing_columns(df, candidates):
    return [c for c in candidates if c in df.columns]


def get_cell_topic_df(cobj, model):
    """
    Return cell-topic matrix as DataFrame:
    rows = cells
    cols = Topic1...TopicN
    """

    df = model.cell_topic.copy()   # (topics × cells)
    print("[cell-topic] use model.cell_topic")

    #  cells × topics
    df = df.T
    df.index = cobj.cell_data.index
    n_topics = model.n_topic
    df.columns = [f"Topic{i+1}" for i in range(n_topics)]

    print("[cell-topic] final shape:", df.shape)

    return df

def export_celltype_topic_tables(tag: str, cobj, model, label_col: str):
    print(f"[export] building celltype-topic tables for {tag}")

    cell_topic_df = get_cell_topic_df(cobj, model)

    meta = cobj.cell_data[[label_col]].copy()
    meta = meta.loc[meta[label_col].notna()].copy()

    merged = meta.join(cell_topic_df, how="inner")

    topic_cols = [c for c in merged.columns if c.startswith("Topic")]
    print(f"[export] merged shape = {merged.shape}")
    print(f"[export] n annotated cells = {merged.shape[0]}")
    print(f"[export] n topics = {len(topic_cols)}")

    # 1) mean topic per cell type
    celltype_topic_mean = merged.groupby(label_col)[topic_cols].mean()
    celltype_topic_mean.to_csv(
        TABDIR / f"{tag}_celltype_topic_mean.tsv",
        sep="\t"
    )
    print("[save]", TABDIR / f"{tag}_celltype_topic_mean.tsv")

    # 2) cell counts per label
    celltype_counts = merged[label_col].value_counts().rename_axis(label_col).reset_index(name="n_cells")
    celltype_counts.to_csv(
        TABDIR / f"{tag}_celltype_counts.tsv",
        sep="\t",
        index=False
    )
    print("[save]", TABDIR / f"{tag}_celltype_counts.tsv")

    # 3) top cell type per topic
    rows = []
    for topic in topic_cols:
        s = celltype_topic_mean[topic].sort_values(ascending=False)
        top1_label = s.index[0]
        top1_score = float(s.iloc[0])

        if len(s) > 1:
            top2_label = s.index[1]
            top2_score = float(s.iloc[1])
        else:
            top2_label = None
            top2_score = np.nan

        specificity_ratio = top1_score / (top2_score + 1e-12) if pd.notna(top2_score) else np.nan

        rows.append({
            "topic": topic,
            "top1_celltype": top1_label,
            "top1_score": top1_score,
            "top2_celltype": top2_label,
            "top2_score": top2_score,
            "specificity_ratio_top1_over_top2": specificity_ratio,
        })

    topic_top_celltype = pd.DataFrame(rows).sort_values(
        by=["specificity_ratio_top1_over_top2", "top1_score"],
        ascending=[False, False]
    )
    topic_top_celltype.to_csv(
        TABDIR / f"{tag}_topic_top_celltype.tsv",
        sep="\t",
        index=False
    )
    print("[save]", TABDIR / f"{tag}_topic_top_celltype.tsv")


def run_one(tag: str, cobj_path: Path, model_path: Path, label_col: str):
    print(f"\n===== [{tag}] =====")
    print("[load] cobj :", cobj_path)
    print("[load] model:", model_path)

    cobj = load_pickle(cobj_path)
    model = load_pickle(model_path)
    cobj.selected_model = model

    print("[model] n_topic =", model.n_topic)
    print("[cells] n =", cobj.cell_data.shape[0])
    print("[meta cols] =", list(cobj.cell_data.columns))

    if label_col in cobj.cell_data.columns:
        print(f"[label counts: {label_col}]")
        print(cobj.cell_data[label_col].value_counts(dropna=False).head(50))

    # clustering
    find_clusters(
        cobj,
        target="cell",
        k=10,
        res=[0.6, 1.2, 3],
        prefix="pycisTopic_",
        scale=True,
        split_pattern="-",
    )

    # UMAP / TSNE
    run_umap(
        cobj,
        target="cell",
        scale=True,
    )

    run_tsne(
        cobj,
        target="cell",
        scale=True,
    )

    cluster_cols = [
        "pycisTopic_leiden_10_0.6",
        "pycisTopic_leiden_10_1.2",
        "pycisTopic_leiden_10_3",
    ]

    # final_class + cluster
    discrete_vars = []
    if label_col in cobj.cell_data.columns and cobj.cell_data[label_col].notna().any():
        discrete_vars.append(label_col)
    discrete_vars += [c for c in cluster_cols if c in cobj.cell_data.columns]

    if discrete_vars:
        plot_metadata(
            cobj,
            reduction_name="UMAP",
            variables=discrete_vars,
            target="cell",
            num_columns=min(4, len(discrete_vars)),
            text_size=10,
            dot_size=0.5,
        )
        save_current(FIGDIR / f"{tag}_UMAP_discrete.pdf")

    cont_candidates = [
        "log10_unique_fragments_count",
        "tss_enrichment",
        "fraction_of_fragments_in_peaks",
        "Doublet_scores_fragments",
        "unique_fragments_count",
        "FRIP",
    ]
    cont_vars = pick_existing_columns(cobj.cell_data, cont_candidates)

    if cont_vars:
        plot_metadata(
            cobj,
            reduction_name="UMAP",
            variables=cont_vars,
            target="cell",
            num_columns=min(4, len(cont_vars)),
            text_size=10,
            dot_size=0.5,
        )
        save_current(FIGDIR / f"{tag}_UMAP_qc.pdf")

    if discrete_vars:
        plot_metadata(
            cobj,
            reduction_name="tSNE",
            variables=discrete_vars,
            target="cell",
            num_columns=min(4, len(discrete_vars)),
            text_size=10,
            dot_size=0.5,
        )
        save_current(FIGDIR / f"{tag}_TSNE_discrete.pdf")

    # topic UMAP
    topic_ids = list(range(1, model.n_topic + 1))
    
    for page_i, topic_chunk in enumerate(chunks(topic_ids, 10), start=1):
        print(f"[plot topic UMAP] {tag} page {page_i}: {topic_chunk}")
    
        plot_topic(
            cobj,
            reduction_name="UMAP",
            target="cell",
            selected_topics=topic_chunk,
            num_columns=5,
            dot_size=0.5,
            alpha=0.8,
            scale=True,
        )
    
        save_current(
            FIGDIR / f"{tag}_UMAP_topics_page{page_i:02d}.png",
            dpi=180,
        )

    # cell-topic heatmap
    plot_celltype_topic_mean_heatmap(tag, cobj, model, label_col)

    # new exports
    if label_col in cobj.cell_data.columns:
        export_celltype_topic_tables(tag, cobj, model, label_col)

    # save
    out_pkl = OUT / f"4_cobj_{tag}_clustvis.pkl"
    with open(out_pkl, "wb") as f:
        pickle.dump(cobj, f)
    print("[save] updated cobj ->", out_pkl)


def main():
    for tag, cfg in DATASETS.items():
        run_one(tag, cfg["cobj"], cfg["model"], cfg["label_col"])


if __name__ == "__main__":
    main()