from __future__ import annotations
import os
os.environ["KERAS_BACKEND"] = "torch"

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# =========================
# paths
# =========================
IN_DIR = Path(
    "/mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/pjt/runs/out/har_integrated"
)

LONG_CSV = IN_DIR / "master_har_3species_2models_long.csv"
WIDE_CSV = IN_DIR / "master_har_3species_2models_wide.csv"
CELLTYPE_COUNT_CSV = IN_DIR / "summary_counts_by_best_celltype_by_mean.csv"

OUT_DIR = IN_DIR / "plots_global"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_N = 50


# =========================
# helpers
# =========================
def save_fig(fig: plt.Figure, out_base: Path, dpi: int = 300) -> None:
    fig.savefig(str(out_base) + ".png", dpi=dpi, bbox_inches="tight")
    fig.savefig(str(out_base) + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved:", str(out_base) + ".png")
    print("Saved:", str(out_base) + ".pdf")


def clean_species_order(values: list[str]) -> list[str]:
    desired = ["human", "chimp", "macaque"]
    present = [x for x in desired if x in values]
    rest = [x for x in values if x not in present]
    return present + sorted(rest)


def clean_model_order(values: list[str]) -> list[str]:
    desired = ["human_model", "macaque_model"]
    present = [x for x in desired if x in values]
    rest = [x for x in values if x not in present]
    return present + sorted(rest)


def get_cellscore_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if str(c).startswith("cellscore__")]
    return cols


def short_celltype_name(col: str) -> str:
    return str(col).replace("cellscore__", "")


def make_boxplot_top_score(long_df: pd.DataFrame) -> None:
    species_order = clean_species_order(sorted(long_df["species"].dropna().unique().tolist()))
    model_order = clean_model_order(sorted(long_df["model"].dropna().unique().tolist()))

    fig, axes = plt.subplots(
        1, len(model_order), figsize=(6 * len(model_order), 5), squeeze=False
    )
    axes = axes[0]

    for ax, model in zip(axes, model_order):
        sub = long_df.loc[long_df["model"] == model].copy()

        data = [
            sub.loc[sub["species"] == sp, "top_score"].dropna().to_numpy()
            for sp in species_order
        ]

        ax.boxplot(data, labels=species_order, showfliers=False)
        ax.set_title(model)
        ax.set_xlabel("species")
        ax.set_ylabel("top_score")
        ax.tick_params(axis="x", rotation=30)

    fig.suptitle("Top score distribution across species and models")
    save_fig(fig, OUT_DIR / "01_top_score_boxplot_by_species_and_model")


def make_violin_top_score(long_df: pd.DataFrame) -> None:
    species_order = clean_species_order(sorted(long_df["species"].dropna().unique().tolist()))
    model_order = clean_model_order(sorted(long_df["model"].dropna().unique().tolist()))

    fig, axes = plt.subplots(
        1, len(model_order), figsize=(6 * len(model_order), 5), squeeze=False
    )
    axes = axes[0]

    for ax, model in zip(axes, model_order):
        sub = long_df.loc[long_df["model"] == model].copy()
        data = [
            sub.loc[sub["species"] == sp, "top_score"].dropna().to_numpy()
            for sp in species_order
        ]

        vp = ax.violinplot(data, showmeans=True, showmedians=False, showextrema=False)
        ax.set_xticks(np.arange(1, len(species_order) + 1))
        ax.set_xticklabels(species_order, rotation=30)
        ax.set_title(model)
        ax.set_xlabel("species")
        ax.set_ylabel("top_score")

    fig.suptitle("Top score violin plot across species and models")
    save_fig(fig, OUT_DIR / "02_top_score_violin_by_species_and_model")


def make_best_celltype_stacked_bar() -> None:
    df = pd.read_csv(CELLTYPE_COUNT_CSV)
    df["model"] = df["model"].astype(str)
    df["species"] = df["species"].astype(str)
    df["best_celltype_by_mean"] = df["best_celltype_by_mean"].astype(str)

    model_order = clean_model_order(sorted(df["model"].dropna().unique().tolist()))
    species_order = clean_species_order(sorted(df["species"].dropna().unique().tolist()))

    all_celltypes = sorted(df["best_celltype_by_mean"].dropna().unique().tolist())

    fig, axes = plt.subplots(
        1, len(model_order), figsize=(7 * len(model_order), 6), squeeze=False
    )
    axes = axes[0]

    for ax, model in zip(axes, model_order):
        sub = df.loc[df["model"] == model].copy()

        pivot = sub.pivot_table(
            index="species",
            columns="best_celltype_by_mean",
            values="n",
            aggfunc="sum",
            fill_value=0,
        )

        for sp in species_order:
            if sp not in pivot.index:
                pivot.loc[sp] = 0

        for ct in all_celltypes:
            if ct not in pivot.columns:
                pivot[ct] = 0

        pivot = pivot.loc[species_order, all_celltypes]

        bottom = np.zeros(len(pivot), dtype=float)
        x = np.arange(len(pivot.index))

        for ct in pivot.columns:
            vals = pivot[ct].to_numpy(dtype=float)
            ax.bar(x, vals, bottom=bottom, label=ct)
            bottom += vals

        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index, rotation=30)
        ax.set_title(model)
        ax.set_xlabel("species")
        ax.set_ylabel("count")

    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(1.02, 1.0), loc="upper left", frameon=False)
    fig.suptitle("Best cell-type composition across species and models")
    save_fig(fig, OUT_DIR / "03_best_celltype_stacked_bar")


def make_scatter_human_vs_macaque_model(wide_df: pd.DataFrame) -> None:
    required = ["top_score__human_model", "top_score__macaque_model", "species"]
    for c in required:
        if c not in wide_df.columns:
            print(f"Skip scatter: missing column {c}")
            return

    fig, ax = plt.subplots(figsize=(6.5, 6))

    species_order = clean_species_order(sorted(wide_df["species"].dropna().unique().tolist()))
    markers = {"human": "o", "chimp": "^", "macaque": "s"}

    all_x = wide_df["top_score__human_model"].to_numpy(dtype=float)
    all_y = wide_df["top_score__macaque_model"].to_numpy(dtype=float)
    finite_mask = np.isfinite(all_x) & np.isfinite(all_y)
    all_x = all_x[finite_mask]
    all_y = all_y[finite_mask]

    if len(all_x) == 0:
        print("Skip scatter: no finite values")
        return

    for sp in species_order:
        sub = wide_df.loc[wide_df["species"] == sp].copy()
        x = sub["top_score__human_model"].to_numpy(dtype=float)
        y = sub["top_score__macaque_model"].to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]

        ax.scatter(
            x,
            y,
            alpha=0.6,
            s=18,
            marker=markers.get(sp, "o"),
            label=sp,
        )

    mn = min(all_x.min(), all_y.min())
    mx = max(all_x.max(), all_y.max())
    ax.plot([mn, mx], [mn, mx], linestyle="--", linewidth=1)

    corr = np.corrcoef(all_x, all_y)[0, 1]

    ax.set_xlabel("top_score (human_model)")
    ax.set_ylabel("top_score (macaque_model)")
    ax.set_title(f"Cross-model comparison of HAR top scores\nPearson r = {corr:.3f}")
    ax.legend(frameon=False)

    save_fig(fig, OUT_DIR / "04_scatter_top_score_human_vs_macaque_model")


def make_scatter_best_celltype_score(wide_df: pd.DataFrame) -> None:
    required = [
        "best_celltype_score__human_model",
        "best_celltype_score__macaque_model",
        "species",
    ]
    for c in required:
        if c not in wide_df.columns:
            print(f"Skip best_celltype_score scatter: missing column {c}")
            return

    fig, ax = plt.subplots(figsize=(6.5, 6))

    species_order = clean_species_order(sorted(wide_df["species"].dropna().unique().tolist()))
    markers = {"human": "o", "chimp": "^", "macaque": "s"}

    all_x = wide_df["best_celltype_score__human_model"].to_numpy(dtype=float)
    all_y = wide_df["best_celltype_score__macaque_model"].to_numpy(dtype=float)
    finite_mask = np.isfinite(all_x) & np.isfinite(all_y)
    all_x = all_x[finite_mask]
    all_y = all_y[finite_mask]

    if len(all_x) == 0:
        print("Skip scatter: no finite values")
        return

    for sp in species_order:
        sub = wide_df.loc[wide_df["species"] == sp].copy()
        x = sub["best_celltype_score__human_model"].to_numpy(dtype=float)
        y = sub["best_celltype_score__macaque_model"].to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]

        ax.scatter(
            x,
            y,
            alpha=0.6,
            s=18,
            marker=markers.get(sp, "o"),
            label=sp,
        )

    mn = min(all_x.min(), all_y.min())
    mx = max(all_x.max(), all_y.max())
    ax.plot([mn, mx], [mn, mx], linestyle="--", linewidth=1)

    corr = np.corrcoef(all_x, all_y)[0, 1]

    ax.set_xlabel("best_celltype_score (human_model)")
    ax.set_ylabel("best_celltype_score (macaque_model)")
    ax.set_title(f"Cross-model comparison of HAR best cell-type scores\nPearson r = {corr:.3f}")
    ax.legend(frameon=False)

    save_fig(fig, OUT_DIR / "05_scatter_best_celltype_score_human_vs_macaque_model")


def select_top_hars_for_heatmap(long_df: pd.DataFrame, top_n: int = 20) -> list[str]:
    tmp = (
        long_df.groupby("har_id", as_index=False)["combined_score"]
        .max()
        .sort_values("combined_score", ascending=False)
        .head(top_n)
    )
    har_ids = tmp["har_id"].astype(str).tolist()
    return har_ids


def make_heatmap_top_hars_cellscore(long_df: pd.DataFrame, model_label: str, top_n: int = 20) -> None:
    cellscore_cols = get_cellscore_cols(long_df)
    if len(cellscore_cols) == 0:
        print(f"Skip heatmap for {model_label}: no cellscore__ columns found")
        return

    sub = long_df.loc[long_df["model"] == model_label].copy()
    if sub.empty:
        print(f"Skip heatmap for {model_label}: empty subset")
        return

    top_hars = select_top_hars_for_heatmap(long_df, top_n=top_n)
    sub = sub.loc[sub["har_id"].astype(str).isin(top_hars)].copy()

    sub = sub.sort_values("combined_score", ascending=False).drop_duplicates("har_id", keep="first")
    if sub.empty:
        print(f"Skip heatmap for {model_label}: no rows after dedup")
        return

    sub["har_label"] = (
        sub["har_id"].astype(str)
        + " | "
        + sub["species"].astype(str)
        + " | "
        + sub["best_celltype_by_mean"].astype(str)
    )

    mat = sub[cellscore_cols].to_numpy(dtype=float)
    row_labels = sub["har_label"].tolist()
    col_labels = [short_celltype_name(c) for c in cellscore_cols]

    order = np.argsort(-sub["best_celltype_score"].to_numpy(dtype=float))
    mat = mat[order, :]
    row_labels = [row_labels[i] for i in order]

    fig_w = max(10, len(col_labels) * 0.6)
    fig_h = max(8, len(row_labels) * 0.35)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(mat, aspect="auto", interpolation="nearest")
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=90)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title(f"Top {top_n} HAR cell-type score heatmap ({model_label})")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("cell-type score")

    save_fig(fig, OUT_DIR / f"06_heatmap_top{top_n}_cellscore__{model_label}")


def make_heatmap_top_hars_topscore(long_df: pd.DataFrame, top_n: int = 20) -> None:
    top_hars = select_top_hars_for_heatmap(long_df, top_n=top_n)
    sub = long_df.loc[long_df["har_id"].astype(str).isin(top_hars)].copy()
    if sub.empty:
        print("Skip top_score heatmap: empty subset")
        return

    sub["col_group"] = sub["species"].astype(str) + "|" + sub["model"].astype(str)

    pivot = sub.pivot_table(
        index="har_id",
        columns="col_group",
        values="top_score",
        aggfunc="max",
    )

    species_order = clean_species_order(sorted(long_df["species"].dropna().unique().tolist()))
    model_order = clean_model_order(sorted(long_df["model"].dropna().unique().tolist()))
    desired_cols = [f"{sp}|{md}" for sp in species_order for md in model_order]
    desired_cols = [c for c in desired_cols if c in pivot.columns]

    pivot = pivot.reindex(columns=desired_cols)

    # max top_score 
    row_order = pivot.max(axis=1).sort_values(ascending=False).index.tolist()
    pivot = pivot.loc[row_order]

    fig_w = max(7, len(pivot.columns) * 1.2)
    fig_h = max(8, len(pivot.index) * 0.35)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", interpolation="nearest")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index.tolist())
    ax.set_title(f"Top {top_n} HAR top_score heatmap across species and models")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("top_score")

    save_fig(fig, OUT_DIR / f"07_heatmap_top{top_n}_top_score_cross_species_models")


def make_delta_histogram(wide_df: pd.DataFrame) -> None:
    col = "delta_top_score_human_minus_macaque_model"
    if col not in wide_df.columns:
        print(f"Skip delta histogram: missing {col}")
        return

    species_order = clean_species_order(sorted(wide_df["species"].dropna().unique().tolist()))
    fig, axes = plt.subplots(
        1, len(species_order), figsize=(5 * len(species_order), 4.5), squeeze=False
    )
    axes = axes[0]

    for ax, sp in zip(axes, species_order):
        sub = wide_df.loc[wide_df["species"] == sp, col].dropna().to_numpy(dtype=float)
        if len(sub) == 0:
            continue

        ax.hist(sub, bins=30)
        ax.axvline(0, linestyle="--", linewidth=1)
        ax.set_title(sp)
        ax.set_xlabel("top_score(human_model) - top_score(macaque_model)")
        ax.set_ylabel("count")

    fig.suptitle("Distribution of cross-model score differences")
    save_fig(fig, OUT_DIR / "08_hist_delta_top_score_human_minus_macaque_model")


def main():
    print("========== HAR global plots ==========")
    print("Reading:", LONG_CSV)
    print("Reading:", WIDE_CSV)

    long_df = pd.read_csv(LONG_CSV)
    wide_df = pd.read_csv(WIDE_CSV)

    print("Long shape:", long_df.shape)
    print("Wide shape:", wide_df.shape)

    make_boxplot_top_score(long_df)
    make_violin_top_score(long_df)
    make_best_celltype_stacked_bar()
    make_scatter_human_vs_macaque_model(wide_df)
    make_scatter_best_celltype_score(wide_df)
    make_heatmap_top_hars_cellscore(long_df, model_label="human_model", top_n=TOP_N)
    make_heatmap_top_hars_cellscore(long_df, model_label="macaque_model", top_n=TOP_N)
    make_heatmap_top_hars_topscore(long_df, top_n=TOP_N)
    make_delta_histogram(wide_df)

    print("\nDone.")
    print("Plots saved to:", OUT_DIR)


if __name__ == "__main__":
    main()