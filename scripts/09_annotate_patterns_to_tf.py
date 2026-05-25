#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pickle
import subprocess
import numpy as np
import pandas as pd
import crested


BASE = Path("/mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/pjt")
STAGE2 = BASE / "runs/out/stage2_topic_modisco_top1000"
TOMTOM = Path("/mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/miniconda3/envs/meme_env/bin/tomtom")


def get_pwm_from_pattern(pat):
    # 你的 all_patterns.pkl 结构：
    # pat["pattern"]["sequence"] = consensus PWM/PPM

    if isinstance(pat, dict):
        if "pattern" in pat and isinstance(pat["pattern"], dict):
            inner = pat["pattern"]
            if "sequence" in inner:
                pwm = np.asarray(inner["sequence"], dtype=float)
            elif "pwm" in inner:
                pwm = np.asarray(inner["pwm"], dtype=float)
            elif "ppm" in inner:
                pwm = np.asarray(inner["ppm"], dtype=float)
            else:
                raise KeyError(f"No sequence/pwm/ppm in pat['pattern']: {inner.keys()}")

        elif "sequence" in pat:
            pwm = np.asarray(pat["sequence"], dtype=float)

        elif "pwm" in pat:
            pwm = np.asarray(pat["pwm"], dtype=float)

        elif "ppm" in pat:
            pwm = np.asarray(pat["ppm"], dtype=float)

        else:
            raise KeyError(f"No usable PWM fields in pattern dict: {pat.keys()}")

    else:
        for k in ["sequence", "pwm", "ppm"]:
            if hasattr(pat, k):
                pwm = np.asarray(getattr(pat, k), dtype=float)
                break
        else:
            pwm = np.asarray(pat, dtype=float)

    if pwm.ndim != 2:
        raise ValueError(f"Bad PWM ndim: {pwm.shape}")

    if pwm.shape[1] != 4 and pwm.shape[0] == 4:
        pwm = pwm.T

    if pwm.shape[1] != 4:
        raise ValueError(f"Bad PWM shape: {pwm.shape}")

    pwm = np.clip(pwm, 1e-6, None)
    pwm = pwm / pwm.sum(axis=1, keepdims=True)

    return pwm

def write_meme(all_patterns, out_meme):
    with open(out_meme, "w") as f:
        f.write("MEME version 4\n\n")
        f.write("ALPHABET= ACGT\n\n")
        f.write("strands: + -\n\n")
        f.write("Background letter frequencies\n")
        f.write("A 0.25 C 0.25 G 0.25 T 0.25\n\n")

        if isinstance(all_patterns, dict):
            items = all_patterns.items()
        else:
            items = [(f"pattern_cluster_{i}", p) for i, p in enumerate(all_patterns)]

        n = 0
        for name, pat in items:
            name = str(name)
            if not name.startswith("pattern_cluster_"):
                name = f"pattern_cluster_{name}"

            pwm = get_pwm_from_pattern(pat)

            f.write(f"MOTIF {name}\n")
            f.write(f"letter-probability matrix: alength= 4 w= {pwm.shape[0]} nsites= 20 E= 0\n")
            for row in pwm:
                f.write(" ".join(f"{x:.6f}" for x in row) + "\n")
            f.write("\n")
            n += 1

    print(f"[write meme] {out_meme} ({n} patterns)")


def load_motif_tf_table(motif_to_tf_file):
    df = pd.read_csv(motif_to_tf_file, sep="\t")

    id_col = "Motif_name"

    tf_cols = [
        "Human_Direct_annot",
        "Human_Orthology_annot",
        "Cluster_Human_Direct_annot",
        "Cluster_Human_Orthology_annot",
    ]

    mapping = {}

    for _, r in df.iterrows():
        motif_id = str(r[id_col]).strip()

        tfs = []
        for c in tf_cols:
            val = r[c]
            if pd.isna(val):
                continue

            for x in str(val).replace(";", ",").split(","):
                x = x.strip()
                if x and x.lower() not in ["nan", "none", "na"]:
                    tfs.append(x)

        mapping[motif_id] = sorted(set(tfs))

    print("[motif tf table]", motif_to_tf_file)
    print("[motif id col]", id_col)
    print("[tf cols]", tf_cols)
    print("[n motif ids]", len(mapping))

    return mapping


def map_target_to_tfs(target_id, motif_to_tfs):
    target_id = str(target_id)

    candidates = [
        target_id,
        target_id.split()[0],
        target_id.split("_")[0],
        target_id.split(".")[0],
    ]

    for c in candidates:
        if c in motif_to_tfs:
            return motif_to_tfs[c]

    hits = []
    for motif_id, tfs in motif_to_tfs.items():
        if motif_id in target_id or target_id in motif_id:
            hits.extend(tfs)

    return sorted(set(hits))


def annotate_species(species):
    print(f"\n===== {species} =====")

    pattern_pkl = STAGE2 / species / "pattern_matrix" / f"{species}_all_patterns.pkl"
    outdir = STAGE2 / species / "pattern_tf_annotation_tomtom"
    outdir.mkdir(parents=True, exist_ok=True)

    query_meme = outdir / f"{species}_patterns_query.meme"
    tomtom_out = outdir / "tomtom_out"

    with open(pattern_pkl, "rb") as f:
        all_patterns = pickle.load(f)

    meme_db, motif_to_tf_file = crested.get_motif_db()
    motif_to_tfs = load_motif_tf_table(motif_to_tf_file)

    write_meme(all_patterns, query_meme)

    cmd = [
        str(TOMTOM),
        "-oc", str(tomtom_out),
        "-no-ssc",
        "-verbosity", "1",
        "-thresh", "0.05",
        str(query_meme),
        str(meme_db),
    ]

    print("[run]", " ".join(cmd))
    subprocess.run(cmd, check=True)

    tsv = tomtom_out / "tomtom.tsv"
    if not tsv.exists():
        raise FileNotFoundError(tsv)

    raw = pd.read_csv(tsv, sep="\t", comment="#")
    raw_out = outdir / f"{species}_tomtom_raw.tsv"
    raw.to_csv(raw_out, sep="\t", index=False)
    print("[save raw]", raw_out)

    rows = []

    for pattern, sub in raw.groupby("Query_ID"):
        sub = sub.sort_values(["q-value", "p-value"], ascending=True)

        tf_hits = []
        motif_hits = []

        for _, r in sub.head(10).iterrows():
            target = str(r["Target_ID"])
            tfs = map_target_to_tfs(target, motif_to_tfs)

            motif_hits.append(target)
            tf_hits.extend(tfs)

        rows.append({
            "pattern_cluster": pattern,
            "top_motif_matches": ",".join(sorted(set(motif_hits))),
            "tf_candidates": ",".join(sorted(set(tf_hits))),
            "n_tf_candidates": len(set(tf_hits)),
            "best_target_id": sub.iloc[0]["Target_ID"],
            "best_p_value": sub.iloc[0]["p-value"],
            "best_q_value": sub.iloc[0]["q-value"],
        })

    out = outdir / f"{species}_pattern_to_tf_candidates.tsv"
    pd.DataFrame(rows).sort_values("pattern_cluster").to_csv(out, sep="\t", index=False)
    print("[save]", out)


def main():
    for sp in ["human", "macaque"]:
        annotate_species(sp)


if __name__ == "__main__":
    main()