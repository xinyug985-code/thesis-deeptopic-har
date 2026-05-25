import os
from pathlib import Path

# ====== output ======
BASE = Path("/cfs/klemming/projects/supr/snic2022-23-547/xinyu")
RUNS = BASE / "runs"
RUNS.mkdir(parents=True, exist_ok=True)
MANIFEST_OUT = RUNS / "fragments_manifest_macaque.tsv"

PEAKS_IN  = BASE / "consensus_peaks_macaque.bed"
PEAKS_OUT = BASE / "consensus_peaks_macaque_3col.bed"

# ====== fragments list (sample_id, path) ======
FRAGMENTS = [
    ("pcd101_R1", "/cfs/klemming/projects/supr/snic2022-23-547/yiquan/macaque_ATAC/gao/pcd101_R1_prefixed_fragments.tsv.gz"),
    ("pcd112_R1", "/cfs/klemming/projects/supr/snic2022-23-547/yiquan/macaque_ATAC/gao/pcd112_R1_prefixed_fragments.tsv.gz"),
    ("pcd127_R1_L", "/cfs/klemming/projects/supr/snic2022-23-547/yiquan/macaque_ATAC/gao/pcd127_R1_L_prefixed_fragments.tsv.gz"),
    ("pcd127_R2_R", "/cfs/klemming/projects/supr/snic2022-23-547/yiquan/macaque_ATAC/gao/pcd127_R2_R_prefixed_fragments.tsv.gz"),
    ("pcd147_R1", "/cfs/klemming/projects/supr/snic2022-23-547/yiquan/macaque_ATAC/gao/pcd147_R1_prefixed_fragments.tsv.gz"),
    ("pcd155_R1", "/cfs/klemming/projects/supr/snic2022-23-547/yiquan/macaque_ATAC/gao/pcd155_R1_prefixed_fragments.tsv.gz"),
    ("pcd155_R2", "/cfs/klemming/projects/supr/snic2022-23-547/yiquan/macaque_ATAC/gao/pcd155_R2_prefixed_fragments.tsv.gz"),
    ("pcd87_R1", "/cfs/klemming/projects/supr/snic2022-23-547/yiquan/macaque_ATAC/gao/pcd87_R1_prefixed_fragments.tsv.gz"),
    ("pcd96_R1", "/cfs/klemming/projects/supr/snic2022-23-547/yiquan/macaque_ATAC/gao/pcd96_R1_prefixed_fragments.tsv.gz"),

]

# ======Write manifest ======
with open(MANIFEST_OUT, "w") as f:
    for sid, path in FRAGMENTS:
        if not os.path.exists(path):
            raise RuntimeError(f"Fragments file not found: {path}")
        f.write(f"{sid}\t{path}\n")

print("[ok] fragments_manifest_macaque.tsv written")
print("[info] n_samples:", len(FRAGMENTS))

# ======Convert peaks to 3-column BED ======
if not PEAKS_IN.exists():
    raise RuntimeError(f"Peaks file not found: {PEAKS_IN}")

with open(PEAKS_IN) as fin:
    first_line = fin.readline().strip().split()

if len(first_line) == 3:
    print("[peaks] already 3-column BED")
    if not PEAKS_OUT.exists():
        PEAKS_OUT.write_text(PEAKS_IN.read_text())
else:
    print(f"[peaks] detected {len(first_line)} columns → converting to 3-column BED")
    with open(PEAKS_IN) as fin, open(PEAKS_OUT, "w") as fout:
        for line in fin:
            if line.strip():
                parts = line.strip().split()
                fout.write(f"{parts[0]}\t{parts[1]}\t{parts[2]}\n")

print("[ok] 3-column peaks file ready:", PEAKS_OUT)