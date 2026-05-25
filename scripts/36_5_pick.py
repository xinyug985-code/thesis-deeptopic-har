#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd

# =========================
# paths
# =========================
BASE_DIR = Path("/mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/pjt")
CASE_DIR = BASE_DIR / "runs" / "out" / "har1687_case"
ISM_DIR = CASE_DIR / "36_3_human_focus_ism"

OUT_ALL = ISM_DIR / "36_top_sensitive_sites_all.csv"
OUT_HIGH = ISM_DIR / "36_top_sensitive_sites_high_only.csv"

# =========================
# params
# =========================
TOP_K_PER_WINDOW = 5
MIN_MAX_DROP = 0.02


def parse_summary_filename(path: Path):
    """
    兼容这种文件名：
    HARsv2_1687__humanSeq__human_model__rank1__summary.csv
    """
    name = path.name
    suffix = "__summary.csv"
    if not name.endswith(suffix):
        return None

    # 排除总 summary
    if name == "HARsv2_1687__human_focus_ism_summary.csv":
        return None

    stem = name[:-len(suffix)]
    parts = stem.split("__")

    if len(parts) != 4:
        return None

    har_id = parts[0]
    species_part = parts[1]
    model_label = parts[2]
    rank_part = parts[3]

    if not rank_part.startswith("rank"):
        return None

    try:
        window_rank = int(rank_part.replace("rank", ""))
    except Exception:
        return None

    species_label = species_part.replace("Seq", "").replace("_seq", "").replace("seq", "")
    species_label = species_label.lower()

    return {
        "har_id": har_id,
        "model_label": model_label,
        "window_rank": window_rank,
        "species_label": species_label,
    }


def main():
    summary_files = sorted(ISM_DIR.glob("*__summary.csv"))

    if len(summary_files) == 0:
        raise FileNotFoundError(f"No summary csv files found in: {ISM_DIR}")

    print(f"Found {len(summary_files)} summary-like files")

    all_rows = []

    for f in summary_files:
        meta = parse_summary_filename(f)
        if meta is None:
            print(f"[SKIP] not a per-window summary: {f.name}")
            continue

        df = pd.read_csv(f)

        required_cols = {
            "position_in_window_0based",
            "position_in_window_1based",
            "ref_base",
            "max_score_drop",
            "mean_score_drop",
            "worst_alt_base",
            "worst_alt_score_drop",
        }
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"{f} missing columns: {missing}")

        df = df.copy()
        df["har_id"] = meta["har_id"]
        df["model_label"] = meta["model_label"]
        df["window_rank"] = meta["window_rank"]
        df["species_label"] = meta["species_label"]
        df["source_file"] = f.name

        df_top = (
            df.sort_values("max_score_drop", ascending=False)
              .head(TOP_K_PER_WINDOW)
              .copy()
        )

        df_top["sensitive_rank"] = range(1, len(df_top) + 1)

        all_rows.append(df_top)

        print(
            f"[OK] {meta['har_id']} | {meta['model_label']} | "
            f"rank{meta['window_rank']} | {meta['species_label']} -> kept top {len(df_top)}"
        )

    if len(all_rows) == 0:
        raise ValueError("No valid per-window summary rows collected")

    out_df = pd.concat(all_rows, axis=0, ignore_index=True)

    out_df = out_df.sort_values(
        ["har_id", "model_label", "window_rank", "species_label", "sensitive_rank"]
    ).reset_index(drop=True)

    keep_cols = [
        "har_id",
        "model_label",
        "window_rank",
        "species_label",
        "sensitive_rank",
        "position_in_window_0based",
        "position_in_window_1based",
        "ref_base",
        "max_score_drop",
        "mean_score_drop",
        "worst_alt_base",
        "worst_alt_score_drop",
        "worst_alt_mutated_score",
        "source_file",
    ]
    keep_cols = [c for c in keep_cols if c in out_df.columns]

    out_df.loc[:, keep_cols].to_csv(OUT_ALL, index=False)
    print("Saved:", OUT_ALL)

    high_df = out_df.loc[out_df["max_score_drop"] >= MIN_MAX_DROP].copy()
    high_df = high_df.sort_values(
        ["har_id", "model_label", "window_rank", "species_label", "max_score_drop"],
        ascending=[True, True, True, True, False]
    ).reset_index(drop=True)

    high_df.loc[:, keep_cols].to_csv(OUT_HIGH, index=False)
    print("Saved:", OUT_HIGH)

    print("\nDone.")
    print("Outputs:")
    print(OUT_ALL)
    print(OUT_HIGH)


if __name__ == "__main__":
    main()