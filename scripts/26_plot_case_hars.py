from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# =========================
# paths
# =========================
BASE_DIR = Path("/mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/pjt/runs/out/har_integrated")
LONG_CSV = BASE_DIR / "master_har_3species_2models_long.csv"
CASE_DIR = BASE_DIR / "case_study"

SELECTED_CSV = CASE_DIR / "selected_hars_all_groups.csv"

OUT_DIR = CASE_DIR / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def save_fig(fig: plt.Figure, out_base: Path, dpi: int = 300) -> None:
    fig.savefig(str(out_base) + ".png", dpi=dpi, bbox_inches="tight")
    fig.savefig(str(out_base) + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved:", str(out_base) + ".png")
    print("Saved:", str(out_base) + ".pdf")


def get_cellscore_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if str(c).startswith("cellscore__")]


def short_celltype_name(col: str) -> str:
    return str(col).replace("cellscore__", "")


def sanitize_filename(x: str) -> str:
    x = str(x)
    for ch in [" ", "/", "\\", "|", ":", ";", ",", "(", ")", "[", "]"]:
        x = x.replace(ch, "_")
    return x


def plot_one_har_case(long_df: pd.DataFrame, har_id: str, species: str, group: str) -> None:
    sub = long_df.loc[
        (long_df["har_id"].astype(str) == str(har_id)) &
        (long_df["species"].astype(str) == str(species))
    ].copy()

    if sub.empty:
        print(f"Skip {har_id} ({species}): no rows found")
        return

    if set(sub["model"].astype(str)) != {"human_model", "macaque_model"}:
        print(f"Warning: {har_id} ({species}) does not have both models")
    sub = sub.sort_values("model").copy()

    cellscore_cols = get_cellscore_cols(sub)
    if len(cellscore_cols) == 0:
        print(f"Skip {har_id} ({species}): no cellscore columns")
        return

    model_order = ["human_model", "macaque_model"]
    sub["model"] = pd.Categorical(sub["model"], categories=model_order, ordered=True)
    sub = sub.sort_values("model")

    top_scores = sub["top_score"].to_numpy(dtype=float)
    best_ct_scores = sub["best_celltype_score"].to_numpy(dtype=float)
    best_ct_names = sub["best_celltype_by_mean"].astype(str).tolist()
    models = sub["model"].astype(str).tolist()

    # top 5 celltypes
    topk = 5
    cellscore_data = {}
    for _, row in sub.iterrows():
        vals = row[cellscore_cols].astype(float)
        vals.index = [short_celltype_name(c) for c in cellscore_cols]
        vals = vals.sort_values(ascending=False).head(topk)
        cellscore_data[str(row["model"])] = vals

    union_ct = []
    for m in model_order:
        if m in cellscore_data:
            union_ct.extend(cellscore_data[m].index.tolist())
    union_ct = list(dict.fromkeys(union_ct))

    bar_mat = []
    for m in model_order:
        if m in cellscore_data:
            vals = cellscore_data[m]
            arr = [float(vals.get(ct, 0.0)) for ct in union_ct]
        else:
            arr = [0.0] * len(union_ct)
        bar_mat.append(arr)
    bar_mat = np.asarray(bar_mat, dtype=float)

    fig = plt.figure(figsize=(12, 4.8))

    ax1 = plt.subplot(1, 3, 1)
    ax1.bar(models, top_scores)
    ax1.set_title("Top score")
    ax1.set_ylabel("score")

    ax2 = plt.subplot(1, 3, 2)
    ax2.bar(models, best_ct_scores)
    ax2.set_title("Best cell-type score")
    ax2.set_ylabel("score")
    for i, name in enumerate(best_ct_names):
        ax2.text(i, best_ct_scores[i], name, rotation=90, va="bottom", ha="center", fontsize=8)

    ax3 = plt.subplot(1, 3, 3)
    x = np.arange(len(union_ct))
    width = 0.35
    if len(model_order) >= 1:
        ax3.bar(x - width / 2, bar_mat[0], width=width, label="human_model")
    if len(model_order) >= 2:
        ax3.bar(x + width / 2, bar_mat[1], width=width, label="macaque_model")
    ax3.set_xticks(x)
    ax3.set_xticklabels(union_ct, rotation=60, ha="right")
    ax3.set_title("Top cell-type scores")
    ax3.legend(frameon=False)

    fig.suptitle(f"{har_id} | {species} | {group}")

    out_base = OUT_DIR / f"{group}__{sanitize_filename(har_id)}__{sanitize_filename(species)}"
    save_fig(fig, out_base)


def make_summary_table(long_df: pd.DataFrame, selected_df: pd.DataFrame) -> None:
    merged = long_df.merge(
        selected_df[["har_id", "species", "group"]].drop_duplicates(),
        on=["har_id", "species"],
        how="inner"
    ).copy()

    keep_cols = [
        "group",
        "har_id",
        "species",
        "model",
        "top_topic",
        "top_score",
        "top_celltype",
        "best_celltype_by_mean",
        "best_celltype_score",
        "margin_top1_top2",
        "celltype_margin_top1_top2",
        "combined_score",
    ]
    keep_cols = [c for c in keep_cols if c in merged.columns]
    merged = merged[keep_cols].copy()

    out_csv = CASE_DIR / "selected_hars_case_summary_long.csv"
    merged.to_csv(out_csv, index=False)
    print("Saved:", out_csv)


def main():
    print("Reading:", LONG_CSV)
    print("Reading:", SELECTED_CSV)

    long_df = pd.read_csv(LONG_CSV)
    selected_df = pd.read_csv(SELECTED_CSV)

    print("Long shape:", long_df.shape)
    print("Selected shape:", selected_df.shape)

    make_summary_table(long_df, selected_df)

    selected_unique = (
        selected_df[["har_id", "species", "group"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    for _, row in selected_unique.iterrows():
        plot_one_har_case(
            long_df=long_df,
            har_id=row["har_id"],
            species=row["species"],
            group=row["group"],
        )

    print("\nDone.")
    print("Case plots saved to:", OUT_DIR)


if __name__ == "__main__":
    main()