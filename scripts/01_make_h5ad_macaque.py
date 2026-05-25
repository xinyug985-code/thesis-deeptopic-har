import os
from pathlib import Path

import anndata as ad
import crested
import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


BASE = Path("/mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/pjt")
DATA = BASE / "data"
OUTD = BASE / "runs" / "out"
OUTD.mkdir(parents=True, exist_ok=True)

GENOME_FA = "/mimer/NOBACKUP/groups/naiss2025-22-612/yiquan/macaque/rheMac10.fa"

DATASETS = {
    "macaque": {
        "beds_folder": str(DATA / "topic_beds_macaque_otsu"),
        "regions_file": str(DATA / "regions_full_macaque.bed"),
        "out_h5ad": str(OUTD / "macaque_topics.h5ad"),
    },
}


def make_one(tag: str, beds_folder: str, regions_file: str, out_h5ad: str) -> None:
    print(f"\n===== [{tag}] =====")
    print("beds_folder:", beds_folder)
    print("regions_file:", regions_file)
    print("out_h5ad:", out_h5ad)

    genome = crested.Genome(GENOME_FA)
    crested.register_genome(genome)
    print("Genome sanity check:", genome.fetch("chr1", 10_000_000, 10_000_010))

    adata = crested.import_beds(
        beds_folder=beds_folder,
        regions_file=regions_file,
    )
    print("Raw adata shape (topics x regions):", adata.shape)

    crested.pp.change_regions_width(adata, width=500)

    crested.pp.train_val_test_split(
        adata,
        strategy="chr",
        val_chroms=["chr8", "chr9"],
        test_chroms=["chr10", "chr18", "chr20"],
    )
    print("Split counts:\n", adata.var["split"].value_counts())

    if "file_path" in adata.obs.columns:
        adata.obs["file_path"] = adata.obs["file_path"].astype(str)

    adata.write_h5ad(out_h5ad)
    print("[OK] saved", out_h5ad)


def main():
    for tag, cfg in DATASETS.items():
        make_one(tag, **cfg)


if __name__ == "__main__":
    main()