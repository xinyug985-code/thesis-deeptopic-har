import os
from pathlib import Path
os.environ["KERAS_BACKEND"] = "torch"   

import keras
import anndata as ad
import crested


BASE = Path("/mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/pjt")
OUTD = BASE / "runs" / "out"
RUN_DIR = OUTD / "evaluation"
RUN_DIR.mkdir(parents=True, exist_ok=True)

H5AD = OUTD / "human_topics.h5ad"
GENOME_FA = "/mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/genomes/hg38/hg38.fa"
MODEL = OUTD / "deeptopic_human" / "final_model.keras"

def main():
    adata = ad.read_h5ad(str(H5AD))
    print("adata shape:", adata.shape)

    genome = crested.Genome(GENOME_FA)
    crested.register_genome(genome)
    print("Genome sanity check:", genome.fetch("chr1", 10_000_000, 10_000_010))
    print("adata:", adata.shape)

    datamodule = crested.tl.data.AnnDataModule(
    adata,
    batch_size=256,  # lower this if you encounter OOM errors
    )

    # load an existing model
    evaluator = crested.tl.Crested(data=datamodule)
    model_path = MODEL

    evaluator.load_model(
        model_path,
        compile=True,
    )

    # evaluate the model on the test set
    evaluator.test()

if __name__ == "__main__":
    main()



