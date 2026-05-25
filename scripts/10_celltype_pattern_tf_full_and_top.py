#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse
import pandas as pd
import numpy as np


BASE = Path("/mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/pjt")
STAGE2 = BASE / "runs/out/stage2_topic_modisco_top1000"


def norm_pattern(x):
    x = str(x).strip()
    if x.startswith("pattern_cluster_"):
        return x
    try:
        return f"pattern_cluster_{int(float(x))}"
    except Exception:
        return x


def simplify_tf_name(tf):
    tf = str(tf).strip()
    if tf == "" or tf.lower() in ["nan", "none", "na"]:
        return None

    up = tf.upper()

    family_rules = [
        # neural / developmental
        ("ASCL", "bHLH_family"),
        ("ATOH", "bHLH_family"),
        ("NEUROD", "bHLH_family"),
        ("NEUROG", "bHLH_family"),
        ("BHLH", "bHLH_family"),
        ("TCF3", "bHLH_family"),
        ("TCF4", "bHLH_family"),

        ("SOX", "SOX_family"),
        ("POU", "POU_family"),
        ("PAX", "PAX_family"),
        ("DLX", "DLX_family"),
        ("LHX", "LHX_family"),
        ("EOMES", "TBOX_family"),
        ("TBX", "TBOX_family"),
        ("TBR", "TBOX_family"),
        ("EMX", "Homeobox_family"),
        ("OTX", "Homeobox_family"),
        ("HOX", "Homeobox_family"),
        ("NKX", "Homeobox_family"),
        ("BARHL", "Homeobox_family"),

        ("MEF2", "MEF2_family"),
        ("FOX", "FOX_family"),
        ("GATA", "GATA_family"),
        ("RFX", "RFX_family"),
        ("TEAD", "TEAD_family"),

        # immune / microglia / ETS
        ("SPI", "ETS/SPI_family"),
        ("PU.1", "ETS/SPI_family"),
        ("ETS", "ETS/SPI_family"),
        ("ELF", "ETS/SPI_family"),
        ("ERG", "ETS/SPI_family"),
        ("FLI", "ETS/SPI_family"),
        ("IRF", "IRF_family"),
        ("RUNX", "RUNX_family"),
        ("NFKB", "NFkB_family"),
        ("REL", "NFkB_family"),

        # broad promoter / GC-rich / zinc finger
        ("KLF", "KLF/SP_family"),
        ("SP", "KLF/SP_family"),
        ("MAZ", "KLF/SP_family"),
        ("WT1", "KLF/SP_family"),
        ("EGR", "EGR_family"),
        ("ZNF", "ZNF_family"),
        ("ZBTB", "ZBTB_family"),

        # chromatin / architectural / general
        ("CTCF", "CTCF"),
        ("BORIS", "CTCF"),
        ("NFIX", "NFI_family"),
        ("NFIA", "NFI_family"),
        ("NFIB", "NFI_family"),
        ("NFIC", "NFI_family"),

        # AP-1 / activity / stress
        ("FOS", "AP1_family"),
        ("JUN", "AP1_family"),
        ("BATF", "AP1_family"),
        ("ATF", "ATF/CREB_family"),
        ("CREB", "ATF/CREB_family"),

        # nuclear receptors / others
        ("NR", "Nuclear_receptor_family"),
        ("RARA", "Nuclear_receptor_family"),
        ("RXR", "Nuclear_receptor_family"),
        ("PPAR", "Nuclear_receptor_family"),
        ("ESR", "Nuclear_receptor_family"),
    ]

    for key, family in family_rules:
        if key in up:
            return family

    import re
    m = re.match(r"([A-Z]+)", up)
    if m:
        prefix = m.group(1)
        if len(prefix) >= 3:
            return f"{prefix}_family"

    return tf


def summarize_tf_modules(tf_str):
    if pd.isna(tf_str):
        return ""

    out = []
    for tf in str(tf_str).replace(";", ",").split(","):
        x = simplify_tf_name(tf)
        if x:
            out.append(x)

    return ",".join(sorted(set(out)))


def read_matrix(matrix_file):
    df = pd.read_csv(matrix_file, sep="\t")

    first = df.columns[0]
    if first.lower() in ["celltype", "cell_type", "index"] or first.startswith("Unnamed"):
        df = df.rename(columns={first: "celltype"})
        df = df.set_index("celltype")
    else:
        df = pd.read_csv(matrix_file, sep="\t", index_col=0)

    df.index = df.index.astype(str)
    df.columns = [norm_pattern(c) for c in df.columns]

    return df

def get_dominant_module(modules):
    priority = [
        "bHLH_family",
        "SOX_family",
        "POU_family",
        "DLX_family",
        "LHX_family",
        "MEF2_family",
        "ETS/SPI_family",
        "IRF_family",
        "KLF/SP_family",
        "FOX_family",
        "PAX_family",
        "NFI_family",
        "CTCF",
        "AP1_family",
        "Homeobox_family",
        "TBOX_family",
        "Nuclear_receptor_family",
        "ZNF_family",
    ]

    modules = [m for m in modules if m and str(m).lower() != "nan"]

    for p in priority:
        if p in modules:
            return p

    return modules[0] if len(modules) > 0 else ""


def infer_biology_label(dominant_module, dominant_celltype):
    ct = str(dominant_celltype).lower()

    if dominant_module == "bHLH_family":
        if any(x in ct for x in ["exneu", "neuroblast", "exipc"]):
            return "excitatory_neuronal_differentiation"
        if any(x in ct for x in ["mge", "cge", "interneuron", "lge"]):
            return "interneuron_or_msn_neurogenic_program"
        return "neurogenic_bHLH_program"

    if dominant_module in ["SOX_family", "POU_family", "PAX_family"]:
        if any(x in ct for x in ["rg", "ipc", "progenitor"]):
            return "progenitor_developmental_program"
        return "developmental_progenitor_like_program"

    if dominant_module in ["DLX_family", "LHX_family", "Homeobox_family"]:
        if any(x in ct for x in ["mge", "cge", "interneuron", "lge"]):
            return "interneuron_or_msn_lineage_program"
        return "homeobox_developmental_program"

    if dominant_module == "MEF2_family":
        return "neuronal_maturation_or_activity_program"

    if dominant_module in ["ETS/SPI_family", "IRF_family", "RUNX_family", "NFkB_family"]:
        if "microglia" in ct:
            return "microglia_immune_regulatory_program"
        return "immune_or_glial_regulatory_program"

    if dominant_module in ["KLF/SP_family", "ZNF_family", "CTCF", "NFI_family"]:
        return "broad_chromatin_or_architectural_program"

    if dominant_module in ["FOX_family", "Nuclear_receptor_family", "SMAD_family"]:
        return "broad_developmental_signaling_program"

    return "unclassified_or_broad_regulatory_program"

def make_pattern_tf_summary(tf_df, outdir, species, agg):
    rows = []

    for _, r in tf_df.iterrows():
        pattern = r["pattern_cluster"]
        tf_candidates = r.get("tf_candidates", "")
        tf_modules = r.get("tf_modules", "")

        modules = [
            x.strip()
            for x in str(tf_modules).split(",")
            if x.strip() and x.strip().lower() != "nan"
        ]

        dominant = get_dominant_module(modules)

        rows.append({
            "pattern_cluster": pattern,
            "dominant_module": dominant,
            "tf_modules": ",".join(sorted(set(modules))),
            "n_tf_modules": len(set(modules)),
            "tf_candidates": tf_candidates,
            "n_tf_candidates": r.get("n_tf_candidates", ""),
            "best_target_id": r.get("best_target_id", ""),
            "best_q_value": r.get("best_q_value", ""),
        })

    out = outdir / f"{species}_pattern_tf_module_summary_{agg}.tsv"
    pd.DataFrame(rows).sort_values("pattern_cluster").to_csv(out, sep="\t", index=False)
    print("[save pattern TF summary]", out)

def run_species(species, agg="mean", top_n=10):
    print(f"\n===== {species} | {agg} =====")

    matrix_dir_new = STAGE2 / species / "celltype_pattern_matrix2"
    matrix_dir_old = STAGE2 / species / "celltype_pattern_matrix"

    matrix_file_new = matrix_dir_new / f"{species}_celltype_by_pattern_matrix_{agg}.tsv"
    matrix_file_old = matrix_dir_old / f"{species}_celltype_by_pattern_matrix_{agg}.tsv"

    if matrix_file_new.exists():
        matrix_file = matrix_file_new
    elif matrix_file_old.exists():
        matrix_file = matrix_file_old
    else:
        raise FileNotFoundError(f"No matrix file found for {species} {agg}")

    tf_file = (
        STAGE2 / species / "pattern_tf_annotation_tomtom"
        / f"{species}_pattern_to_tf_candidates.tsv"
    )

    outdir = STAGE2 / species / "celltype_pattern_tf_full"
    outdir.mkdir(parents=True, exist_ok=True)

    print("[matrix]", matrix_file)
    print("[tf]", tf_file)

    mat = read_matrix(matrix_file)
    tf_df = pd.read_csv(tf_file, sep="\t")

    tf_df["pattern_cluster"] = tf_df["pattern_cluster"].apply(norm_pattern)
    tf_df["tf_modules"] = tf_df["tf_candidates"].apply(summarize_tf_modules)
    make_pattern_tf_summary(tf_df, outdir, species, agg)

    # full long table
    long_df = (
        mat.reset_index()
        .melt(
            id_vars="celltype",
            var_name="pattern_cluster",
            value_name="importance",
        )
    )

    long_df["abs_importance"] = long_df["importance"].abs()

    merged = long_df.merge(
        tf_df[
            [
                "pattern_cluster",
                "tf_candidates",
                "tf_modules",
                "top_motif_matches",
                "best_target_id",
                "best_p_value",
                "best_q_value",
            ]
        ],
        on="pattern_cluster",
        how="left",
    )

    merged["rank_in_celltype"] = (
        merged
        .groupby("celltype")["abs_importance"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    full_out = outdir / f"{species}_celltype_pattern_tf_long_{agg}.tsv"
    merged.to_csv(full_out, sep="\t", index=False)
    print("[save full]", full_out)

    # top per celltype
    top_df = (
        merged[merged["rank_in_celltype"] <= top_n]
        .sort_values(["celltype", "rank_in_celltype"])
        .copy()
    )

    top_out = outdir / f"{species}_celltype_pattern_tf_top{top_n}_{agg}.tsv"
    top_df.to_csv(top_out, sep="\t", index=False)
    print("[save top]", top_out)

    compat = top_df.rename(columns={
        "rank_in_celltype": "rank"
    })[
        [
            "celltype",
            "pattern_cluster",
            "rank",
            "importance",
            "tf_candidates",
            "tf_modules",
            "best_q_value",
        ]
    ]

    compat_out = outdir / f"{species}_top_patterns_per_celltype_with_tf_{agg}.tsv"
    compat.to_csv(compat_out, sep="\t", index=False)
    print("[save compat]", compat_out)

    # top selected pattern matrix for heatmap / plotting
    selected_patterns = sorted(top_df["pattern_cluster"].unique())
    top_mat = mat[selected_patterns].copy()

    top_matrix_out = outdir / f"{species}_celltype_by_top{top_n}_pattern_matrix_{agg}.tsv"
    top_mat.to_csv(top_matrix_out, sep="\t")
    print("[save top matrix]", top_matrix_out)

    # summary: each celltype has what TF modules
    rows = []
    for ct, sub in top_df.groupby("celltype"):
        modules = []
        for x in sub["tf_modules"]:
            if pd.isna(x):
                continue
            modules.extend([i.strip() for i in str(x).split(",") if i.strip()])

        rows.append({
            "celltype": ct,
            "top_tf_modules": ",".join(sorted(set(modules))),
            "n_top_patterns": sub["pattern_cluster"].nunique(),
            "n_tf_modules": len(set(modules)),
        })

    summary = pd.DataFrame(rows)
    summary_out = outdir / f"{species}_tf_modules_by_celltype_top{top_n}_{agg}.tsv"
    summary.to_csv(summary_out, sep="\t", index=False)
    print("[save summary]", summary_out)
    # =========================================================
    # pattern-centric curated summary:
    # pattern -> dominant celltype -> TF module -> biology label
    # =========================================================
    pattern_rows = []

    for pat, sub in merged.groupby("pattern_cluster"):
        sub2 = sub.sort_values("abs_importance", ascending=False).copy()

        if sub2.empty:
            continue

        dominant_ct = str(sub2.iloc[0]["celltype"])
        dominant_score = float(sub2.iloc[0]["importance"])
        dominant_abs_score = float(sub2.iloc[0]["abs_importance"])

        if sub2.shape[0] > 1:
            second_ct = str(sub2.iloc[1]["celltype"])
            second_score = float(sub2.iloc[1]["importance"])
            second_abs_score = float(sub2.iloc[1]["abs_importance"])
        else:
            second_ct = ""
            second_score = np.nan
            second_abs_score = np.nan

        if second_abs_score and second_abs_score != 0 and not np.isnan(second_abs_score):
            specificity_ratio = dominant_abs_score / second_abs_score
        else:
            specificity_ratio = np.nan

        top3 = sub2.head(3)
        top3_celltypes = ",".join(top3["celltype"].astype(str).tolist())
        top3_importance = ",".join([f"{x:.4g}" for x in top3["importance"].astype(float).tolist()])

        tf_modules_all = []
        for x in sub2["tf_modules"].dropna().unique():
            tf_modules_all.extend([
                i.strip()
                for i in str(x).split(",")
                if i.strip() and i.strip().lower() != "nan"
            ])

        tf_modules_all = sorted(set(tf_modules_all))
        dominant_module = get_dominant_module(tf_modules_all)
        biology_label = infer_biology_label(dominant_module, dominant_ct)

        tf_candidates = ""
        best_q_value = ""
        best_target_id = ""

        first_tf = sub2.dropna(subset=["tf_candidates"]).head(1)
        if not first_tf.empty:
            tf_candidates = first_tf.iloc[0].get("tf_candidates", "")
            best_q_value = first_tf.iloc[0].get("best_q_value", "")
            best_target_id = first_tf.iloc[0].get("best_target_id", "")

        pattern_rows.append({
            "pattern_cluster": pat,
            "dominant_module": dominant_module,
            "biology_label": biology_label,

            "dominant_celltype": dominant_ct,
            "dominant_importance": dominant_score,

            "second_celltype": second_ct,
            "second_importance": second_score,
            "specificity_ratio_top1_over_top2_abs": specificity_ratio,

            "top3_celltypes": top3_celltypes,
            "top3_importance": top3_importance,

            "tf_modules": ",".join(tf_modules_all),
            "n_tf_modules": len(tf_modules_all),
            "tf_candidates": tf_candidates,
            "best_target_id": best_target_id,
            "best_q_value": best_q_value,
        })

    pattern_summary = pd.DataFrame(pattern_rows)

    pattern_summary_out = outdir / f"{species}_pattern_dominant_celltype_tf_summary_{agg}.tsv"
    pattern_summary.sort_values(
        ["biology_label", "dominant_celltype", "pattern_cluster"]
    ).to_csv(pattern_summary_out, sep="\t", index=False)

    print("[save pattern dominant celltype TF summary]", pattern_summary_out)

    # =========================================================
    # biology-label summary:
    # one line per biological program
    # =========================================================
    label_rows = []

    for label, sub in pattern_summary.groupby("biology_label"):
        dominant_celltypes = sorted(set(sub["dominant_celltype"].dropna().astype(str)))
        dominant_modules = sorted(set(sub["dominant_module"].dropna().astype(str)))

        label_rows.append({
            "biology_label": label,
            "n_patterns": sub["pattern_cluster"].nunique(),
            "dominant_modules": ",".join(dominant_modules),
            "representative_patterns": ",".join(sub["pattern_cluster"].head(10).astype(str)),
            "dominant_celltypes": ",".join(dominant_celltypes),
        })

    label_summary = pd.DataFrame(label_rows)

    label_summary_out = outdir / f"{species}_biology_label_summary_dominant_celltype_{agg}.tsv"
    label_summary.sort_values("n_patterns", ascending=False).to_csv(
        label_summary_out,
        sep="\t",
        index=False,
    )

    print("[save biology label summary dominant]", label_summary_out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--species", choices=["human", "macaque", "both"], default="both")
    parser.add_argument("--agg", choices=["mean", "max", "both"], default="both")
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args()

    species_list = ["human", "macaque"] if args.species == "both" else [args.species]
    agg_list = ["mean", "max"] if args.agg == "both" else [args.agg]

    for sp in species_list:
        for agg in agg_list:
            run_species(sp, agg=agg, top_n=args.top_n)


if __name__ == "__main__":
    main()