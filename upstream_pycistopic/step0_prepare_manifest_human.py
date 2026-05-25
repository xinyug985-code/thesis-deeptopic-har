import os
from pathlib import Path

# ====== output ======
BASE = Path("/cfs/klemming/projects/supr/snic2022-23-547/xinyu")
RUNS = BASE / "runs"
RUNS.mkdir(parents=True, exist_ok=True)
MANIFEST_OUT = RUNS / "fragments_manifest_human.tsv"

PEAKS_IN  = BASE / "consensus_peaks_human.bed"
PEAKS_OUT = BASE / "consensus_peaks_human_3col.bed"

# ====== fragments list (sample_id, path) ======
FRAGMENTS = [
    ("hft_w16_p7_r1", "/cfs/klemming/projects/supr/snic2022-23-547/yiquan/human_ATAC/Cell/data/fragment/m_hft_w16_p7_r1_fragments.tsv.gz"),
    ("hft_w16_p7_r2", "/cfs/klemming/projects/supr/snic2022-23-547/yiquan/human_ATAC/Cell/data/fragment/m_hft_w16_p7_r2_fragments.tsv.gz"),
    ("hft_w16_p7_r3", "/cfs/klemming/projects/supr/snic2022-23-547/yiquan/human_ATAC/Cell/data/fragment/m_hft_w16_p7_r3_fragments.tsv.gz"),
    ("hft_w20_p3_r1", "/cfs/klemming/projects/supr/snic2022-23-547/yiquan/human_ATAC/Cell/data/fragment/m_hft_w20_p3_r1_fragments.tsv.gz"),
    ("hft_w20_p3_r2", "/cfs/klemming/projects/supr/snic2022-23-547/yiquan/human_ATAC/Cell/data/fragment/m_hft_w20_p3_r2_fragments.tsv.gz"),
    ("hft_w21_p5_r1", "/cfs/klemming/projects/supr/snic2022-23-547/yiquan/human_ATAC/Cell/data/fragment/m_hft_w21_p5_r1_fragments.tsv.gz"),
    ("hft_w21_p5_r2", "/cfs/klemming/projects/supr/snic2022-23-547/yiquan/human_ATAC/Cell/data/fragment/m_hft_w21_p5_r2_fragments.tsv.gz"),
    ("hft_w21_p5_r3", "/cfs/klemming/projects/supr/snic2022-23-547/yiquan/human_ATAC/Cell/data/fragment/m_hft_w21_p5_r3_fragments.tsv.gz"),
    ("hft_w21_p5_r4", "/cfs/klemming/projects/supr/snic2022-23-547/yiquan/human_ATAC/Cell/data/fragment/m_hft_w21_p5_r4_fragments.tsv.gz"),
    ("hft_w24_p6_r1", "/cfs/klemming/projects/supr/snic2022-23-547/yiquan/human_ATAC/Cell/data/fragment/m_hft_w24_p6_r1_fragments.tsv.gz"),
    ("hft_w24_p6_r2", "/cfs/klemming/projects/supr/snic2022-23-547/yiquan/human_ATAC/Cell/data/fragment/m_hft_w24_p6_r2_fragments.tsv.gz"),
    ("hft_w24_p6_r3", "/cfs/klemming/projects/supr/snic2022-23-547/yiquan/human_ATAC/Cell/data/fragment/m_hft_w24_p6_r3_fragments.tsv.gz"),
    ("hft_w24_p6_r4", "/cfs/klemming/projects/supr/snic2022-23-547/yiquan/human_ATAC/Cell/data/fragment/m_hft_w24_p6_r4_fragments.tsv.gz"),

    ("Cortex_GW17", "/cfs/klemming/projects/supr/snic2022-23-547/yiquan/human_ATAC/Nature/fragment/Cortex_GW17_fragments.tsv.gz"),
    ("Cortex_GW18", "/cfs/klemming/projects/supr/snic2022-23-547/yiquan/human_ATAC/Nature/fragment/Cortex_GW18_fragments.tsv.gz"),
    ("Cortex_GW21", "/cfs/klemming/projects/supr/snic2022-23-547/yiquan/human_ATAC/Nature/fragment/Cortex_GW21_fragments.tsv.gz"),
    ("Insula_GW20", "/cfs/klemming/projects/supr/snic2022-23-547/yiquan/human_ATAC/Nature/fragment/Insula_GW20_fragments.tsv.gz"),
    ("M1_GW20", "/cfs/klemming/projects/supr/snic2022-23-547/yiquan/human_ATAC/Nature/fragment/M1_GW20_fragments.tsv.gz"),
    ("MGE_GW20", "/cfs/klemming/projects/supr/snic2022-23-547/yiquan/human_ATAC/Nature/fragment/MGE_GW20_fragments.tsv.gz"),
    ("Parietal_GW20", "/cfs/klemming/projects/supr/snic2022-23-547/yiquan/human_ATAC/Nature/fragment/Parietal_GW20_fragments.tsv.gz"),
    ("PFC_GW20", "/cfs/klemming/projects/supr/snic2022-23-547/yiquan/human_ATAC/Nature/fragment/PFC_GW20_fragments.tsv.gz"),
    ("Somato_GW20", "/cfs/klemming/projects/supr/snic2022-23-547/yiquan/human_ATAC/Nature/fragment/Somato_GW20_fragments.tsv.gz"),
    ("V1_GW20", "/cfs/klemming/projects/supr/snic2022-23-547/yiquan/human_ATAC/Nature/fragment/V1_GW20_fragments.tsv.gz"),
    ("Temporal_GW20", "/cfs/klemming/projects/supr/snic2022-23-547/yiquan/human_ATAC/Nature/fragment/Temporal_GW20_fragments.tsv.gz"),

    ("HES1", "/cfs/klemming/projects/supr/snic2022-23-547/yiquan/human_ATAC/DC/fragment/m_HES1_fragments.tsv.gz"),
    ("P18861_1001", "/cfs/klemming/projects/supr/snic2022-23-547/yiquan/human_ATAC/DC/fragment/m_P18861_1001_fragments.tsv.gz"),
    ("P18861_1002", "/cfs/klemming/projects/supr/snic2022-23-547/yiquan/human_ATAC/DC/fragment/m_P18861_1002_fragments.tsv.gz"),

    ("10X291_2", "/cfs/klemming/projects/supr/snic2022-23-547/yiquan/human_ATAC/Biox/fragment/m_10X291_2_fragments.tsv.gz"),
    ("10X291_3", "/cfs/klemming/projects/supr/snic2022-23-547/yiquan/human_ATAC/Biox/fragment/m_10X291_3_fragments.tsv.gz"),
    ("10X313_1", "/cfs/klemming/projects/supr/snic2022-23-547/yiquan/human_ATAC/Biox/fragment/m_10X313_1_fragments.tsv.gz"),
    ("10X313_2", "/cfs/klemming/projects/supr/snic2022-23-547/yiquan/human_ATAC/Biox/fragment/m_10X313_2_fragments.tsv.gz"),
    ("10X346_2", "/cfs/klemming/projects/supr/snic2022-23-547/yiquan/human_ATAC/Biox/fragment/m_10X346_2_fragments.tsv.gz"),
    ("10X347_2", "/cfs/klemming/projects/supr/snic2022-23-547/yiquan/human_ATAC/Biox/fragment/m_10X347_2_fragments.tsv.gz"),
    ("10X366_4", "/cfs/klemming/projects/supr/snic2022-23-547/yiquan/human_ATAC/Biox/fragment/m_10X366_4_fragments.tsv.gz"),
    ("10X369_1", "/cfs/klemming/projects/supr/snic2022-23-547/yiquan/human_ATAC/Biox/fragment/m_10X369_1_fragments.tsv.gz"),
    ("10X370_1", "/cfs/klemming/projects/supr/snic2022-23-547/yiquan/human_ATAC/Biox/fragment/m_10X370_1_fragments.tsv.gz"),
    ("10X402_1", "/cfs/klemming/projects/supr/snic2022-23-547/yiquan/human_ATAC/Biox/fragment/m_10X402_1_fragments.tsv.gz"),
    ("10X406_7", "/cfs/klemming/projects/supr/snic2022-23-547/yiquan/human_ATAC/Biox/fragment/m_10X406_7_fragments.tsv.gz"),
    ("10X420_1", "/cfs/klemming/projects/supr/snic2022-23-547/yiquan/human_ATAC/Biox/fragment/m_10X420_1_fragments.tsv.gz"),

    ("4", "/cfs/klemming/projects/supr/snic2022-23-547/yiquan/human_ATAC/Advance/fragment/m_4.tsv.gz"),
    ("8", "/cfs/klemming/projects/supr/snic2022-23-547/yiquan/human_ATAC/Advance/fragment/m_8.tsv.gz"),
    ("11", "/cfs/klemming/projects/supr/snic2022-23-547/yiquan/human_ATAC/Advance/fragment/m_11.tsv.gz"),
    ("16", "/cfs/klemming/projects/supr/snic2022-23-547/yiquan/human_ATAC/Advance/fragment/m_16.tsv.gz"),
]

# ======Write manifest ======
with open(MANIFEST_OUT, "w") as f:
    for sid, path in FRAGMENTS:
        if not os.path.exists(path):
            raise RuntimeError(f"Fragments file not found: {path}")
        f.write(f"{sid}\t{path}\n")

print("[ok] fragments_manifest_human.tsv written")
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