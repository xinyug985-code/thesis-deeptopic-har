import os
import gzip
import pickle
from pathlib import Path

import pandas as pd
from pycisTopic.cistopic_class import create_cistopic_object_from_fragments, merge

# ====== Paths / settings ======
BASE = Path("/cfs/klemming/projects/supr/snic2022-23-547/xinyu")
RUNS = BASE / "runs"
OUTDIR = RUNS / "out"
OUTDIR.mkdir(parents=True, exist_ok=True)

MANIFEST = RUNS / "fragments_manifest_macaque.tsv"
META_TSV = BASE / "metadata_pycistopic_macaque.tsv"
PEAKS_BED = BASE / "consensus_peaks_macaque_3col.bed"

OUT_PKL = OUTDIR / "1_cobj_macaque.pkl"
REPORT_TSV = OUTDIR / "step1_validbc_report_macaque.tsv"
META_FILTERED_OUT = OUTDIR / "metadata.macaque.filtered_noUnknown.tsv"

PROJECT = "xinyu_topicpipe_macaque"
N_CPU = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))

ID_COL = "Unnamed: 0"
CLASS_COL = "annotation"


# ====== Helper: detect delimiter from fragments ======
def detect_delimiter(frag_path: str, sample_id: str) -> str:
    with gzip.open(frag_path, "rt") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                raise RuntimeError(f"Unexpected fragments format in {frag_path}: fewer than 4 columns")
            barcode = parts[3]
            break
        else:
            raise RuntimeError(f"Empty fragments file: {frag_path}")

    if not barcode.startswith(sample_id):
        raise RuntimeError(
            f"Barcode does not start with sample_id.\n"
            f"sample_id={sample_id}\nbarcode={barcode}\nfrag={frag_path}"
        )
    if len(barcode) <= len(sample_id):
        raise RuntimeError(f"Barcode too short: {barcode}")

    return barcode[len(sample_id)]


# ====== Load metadata (only needed cols) ======
meta = pd.read_csv(
    META_TSV,
    sep="\t",
    usecols=[ID_COL, CLASS_COL],
    dtype={ID_COL: "string", CLASS_COL: "string"},
)

# ====== Filter Unknown (keep all remaining cells) ======
cls = meta[CLASS_COL].fillna("unknown").astype("string").str.strip().str.lower()
unknown_set = {"unknown", "na", "nan", ""}

meta_f = meta[~cls.isin(unknown_set)].copy()
meta_f.to_csv(META_FILTERED_OUT, sep="\t", index=False)

print("[meta] total rows:", meta.shape[0])
print("[meta] after Unknown filter:", meta_f.shape[0])
print("[meta] n_classes after filter:", meta_f[CLASS_COL].nunique())
print("[meta] wrote:", META_FILTERED_OUT)

meta_ids = meta_f[ID_COL].astype("string").drop_duplicates()


# ====== Load fragments manifest ======
manifest = pd.read_csv(
    MANIFEST,
    sep="\t",
    header=None,
    names=["sample_id", "fragments"],
)
manifest["sample_id"] = manifest["sample_id"].astype(str)
print("[manifest] n_samples =", manifest.shape[0])

# ====== Create per-sample cobj ======
cobjs = []
report_rows = []

for _, row in manifest.iterrows():
    sid = row["sample_id"]
    frag = row["fragments"]

    if not os.path.exists(frag):
        raise RuntimeError(f"Fragments file missing: {frag}")

    delim = detect_delimiter(frag, sid)
    prefix = f"{sid}{delim}"

    # keep ALL non-Unknown barcodes for this sample
    all_bc = meta_ids.loc[meta_ids.str.startswith(prefix)].drop_duplicates().to_list()
    n_keep = len(all_bc)

    print(f"[{sid}] delimiter={repr(delim)} kept_bc={n_keep}")

    if n_keep == 0:
        raise RuntimeError(
            f"No valid_bc matched for sample {sid}.\n"
            f"Expected prefix: {prefix}\n"
            f"Tip: check delimiter and whether this sample has any non-Unknown final_class cells."
        )

    cobj = create_cistopic_object_from_fragments(
        path_to_fragments=frag,
        path_to_regions=str(PEAKS_BED),
        valid_bc=all_bc,
        n_cpu=N_CPU,
        project=f"{PROJECT}_{sid}",
        split_pattern='-'
    )

    cobjs.append(cobj)
    report_rows.append((sid, frag, delim, n_keep))

# ====== Merge all samples ======
print("[merge] merging", len(cobjs), "cisTopic objects...")
merged = merge(cobjs, project=PROJECT, split_pattern="___")
merged.project = PROJECT

# ====== Save merged object ======
with open(OUT_PKL, "wb") as f:
    pickle.dump(merged, f)
print("[save] wrote:", OUT_PKL)

# ====== Save report ======
with open(REPORT_TSV, "w") as f:
    f.write("sample_id\tfragments\tdelimiter\tvalid_bc_n_after_noUnknown\n")
    for sid, frag, delim, n_keep in report_rows:
        f.write(f"{sid}\t{frag}\t{delim}\t{n_keep}\n")
print("[save] report:", REPORT_TSV)

print("\nStep1 macaque full completed successfully.")