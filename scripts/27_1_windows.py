from __future__ import annotations
import os
os.environ["KERAS_BACKEND"] = "torch"

from pathlib import Path
import pandas as pd
import numpy as np
import crested


# ============================================================
# paths
# ============================================================
BASE_DIR = Path("/mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/pjt")
CASE_DIR = BASE_DIR / "runs" / "out" / "case6_contribution_crossseq"

SUMMARY_CSV = CASE_DIR / "selected6_crossseq_contribution_summary.csv"

HUMAN_FA = Path("/mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/genomes/hg38/hg38.fa")
MACAQUE_FA = Path("/mimer/NOBACKUP/groups/naiss2025-22-612/yiquan/macaque/rheMac10.fa")

OUT_CSV = CASE_DIR / "selected6_top_contribution_windows.csv"
OUT_FASTA = CASE_DIR / "selected6_top_contribution_windows.fa"


# ============================================================
# params
# ============================================================
WINDOW_SIZE = 30
TOP_K = 3
MIN_SEPARATION = 20   


# ============================================================
# helpers
# ============================================================
def sanitize_name(x: str) -> str:
    x = str(x)
    for ch in [" ", "/", "\\", "|", ":", ";", ",", "(", ")", "[", "]"]:
        x = x.replace(ch, "_")
    return x


def parse_region(region: str):
    """
    chr1:100-200 -> chrom, start, end
    """
    chrom, pos = region.split(":")
    start, end = pos.split("-")
    return chrom, int(start), int(end)


def get_genome(seq_species: str):
    seq_species = str(seq_species).lower()
    if seq_species == "human":
        return crested.Genome(HUMAN_FA)
    elif seq_species == "macaque":
        return crested.Genome(MACAQUE_FA)
    else:
        raise ValueError(f"Unsupported seq_species: {seq_species}")


def fetch_seq(seq_species: str, chrom: str, start: int, end: int) -> str:
    genome = get_genome(seq_species)
    return genome.fetch(chrom, int(start), int(end)).upper()


def greedy_top_nonoverlap_windows(
    values: np.ndarray,
    window_size: int = 30,
    top_k: int = 3,
    min_separation: int = 20,
):

    n = len(values)
    if n < window_size:
        return []

    # sliding window sum
    window_scores = []
    current = values[:window_size].sum()
    window_scores.append(current)
    for i in range(1, n - window_size + 1):
        current = current - values[i - 1] + values[i + window_size - 1]
        window_scores.append(current)

    candidates = [
        {"start": i, "end": i + window_size, "score": float(window_scores[i])}
        for i in range(len(window_scores))
    ]
    candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)

    selected = []
    for cand in candidates:
        ok = True
        for s in selected:
            if not (cand["end"] + min_separation <= s["start"] or cand["start"] >= s["end"] + min_separation):
                ok = False
                break
        if ok:
            selected.append(cand)
        if len(selected) >= top_k:
            break
    for rank, s in enumerate(selected, start=1):
        s["rank"] = rank
    return selected


def load_profile_csv(case_dir: Path, group: str, har_id: str, seq_species: str, model_label: str) -> Path:
    prefix = case_dir / f"{group}__{sanitize_name(har_id)}"
    f = prefix.parent / f"{prefix.name}__{seq_species}_seq__{model_label}__profile.csv"
    return f


# ============================================================
# main
# ============================================================
def main():
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(f"Missing summary csv: {SUMMARY_CSV}")

    summary = pd.read_csv(SUMMARY_CSV)
    print("Loaded summary:", SUMMARY_CSV)
    print("Shape:", summary.shape)

    out_rows = []
    fasta_records = []

    for _, row in summary.iterrows():
        group = str(row["group"])
        har_id = str(row["har_id"])
        seq_species = str(row["seq_species"])
        region = str(row["region"])

        chrom, region_start, region_end = parse_region(region)

        print("=" * 80)
        print(f"Processing {har_id} | {group} | {seq_species} | {region}")

        for model_label in ["human_model", "macaque_model"]:
            prof_csv = load_profile_csv(
                case_dir=CASE_DIR,
                group=group,
                har_id=har_id,
                seq_species=seq_species,
                model_label=model_label,
            )

            if not prof_csv.exists():
                print(f"[WARN] Missing profile file: {prof_csv}")
                continue

            prof = pd.read_csv(prof_csv)
            if "abs_contribution" not in prof.columns:
                raise ValueError(f"{prof_csv} missing abs_contribution column")

            abs_vals = prof["abs_contribution"].to_numpy(dtype=float)

            windows = greedy_top_nonoverlap_windows(
                abs_vals,
                window_size=WINDOW_SIZE,
                top_k=TOP_K,
                min_separation=MIN_SEPARATION,
            )

            if len(windows) == 0:
                print(f"[WARN] No windows found for {har_id} | {seq_species} | {model_label}")
                continue

            full_seq = fetch_seq(seq_species, chrom, region_start, region_end)
            if len(full_seq) != (region_end - region_start):
                print(
                    f"[WARN] fetched seq len {len(full_seq)} != expected {region_end - region_start} "
                    f"for {har_id} | {seq_species}"
                )

            top_topic_col = f"{model_label}_top_topic"
            top_score_col = f"{model_label}_top_score"
            best_ct_col = f"{model_label}_best_celltype"

            target_topic = row[top_topic_col] if top_topic_col in row.index else ""
            top_score = row[top_score_col] if top_score_col in row.index else np.nan
            best_ct = row[best_ct_col] if best_ct_col in row.index else ""

            for w in windows:
                start_in_seq = int(w["start"])
                end_in_seq = int(w["end"])

                genomic_start = region_start + start_in_seq
                genomic_end = region_start + end_in_seq

                subseq = full_seq[start_in_seq:end_in_seq]
                max_abs = float(abs_vals[start_in_seq:end_in_seq].max())
                mean_abs = float(abs_vals[start_in_seq:end_in_seq].mean())

                record_id = (
                    f"{har_id}|{group}|{seq_species}|{model_label}|"
                    f"rank{w['rank']}|{chrom}:{genomic_start}-{genomic_end}"
                )

                out_rows.append({
                    "group": group,
                    "har_id": har_id,
                    "seq_species": seq_species,
                    "model_label": model_label,
                    "region": region,
                    "target_topic": target_topic,
                    "top_score": top_score,
                    "best_celltype": best_ct,
                    "window_rank": w["rank"],
                    "window_score_sum_abs": float(w["score"]),
                    "window_max_abs": max_abs,
                    "window_mean_abs": mean_abs,
                    "window_size": WINDOW_SIZE,
                    "start_in_seq_0based": start_in_seq,
                    "end_in_seq_0based_exclusive": end_in_seq,
                    "chrom": chrom,
                    "genomic_start_0based": genomic_start,
                    "genomic_end_0based_exclusive": genomic_end,
                    "sequence": subseq,
                })

                fasta_records.append((record_id, subseq))

    out_df = pd.DataFrame(out_rows)
    out_df = out_df.sort_values(
        ["group", "har_id", "seq_species", "model_label", "window_rank"]
    ).reset_index(drop=True)

    out_df.to_csv(OUT_CSV, index=False)
    print("Saved:", OUT_CSV)

    with open(OUT_FASTA, "w") as f:
        for rec_id, seq in fasta_records:
            f.write(f">{rec_id}\n{seq}\n")
    print("Saved:", OUT_FASTA)

    print("\nDone.")
    print("Main outputs:")
    print(OUT_CSV)
    print(OUT_FASTA)


if __name__ == "__main__":
    main()