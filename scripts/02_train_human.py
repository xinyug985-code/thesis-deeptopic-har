import os
from pathlib import Path

os.environ["KERAS_BACKEND"] = "torch"

import anndata as ad
import crested


BASE = Path("/mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/pjt")
OUTD = BASE / "runs" / "out"
RUN_DIR = OUTD / "deeptopic_human"
RUN_DIR.mkdir(parents=True, exist_ok=True)

H5AD = OUTD / "human_topics.h5ad"
GENOME_FA = "/mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/genomes/hg38/hg38.fa"

SEQ_LEN = 500
BATCH_SIZE = 128
EPOCHS = 100


def _get_trained_model(trainer):
    for name in ("model", "_model", "net", "_net"):
        if hasattr(trainer, name):
            m = getattr(trainer, name)
            if m is not None:
                return m
    return None


def main():
    adata = ad.read_h5ad(str(H5AD))

    genome = crested.Genome(GENOME_FA)
    crested.register_genome(genome)
    print("Genome sanity check:", genome.fetch("chr1", 10_000_000, 10_000_010))
    print("adata:", adata.shape)

    datamodule = crested.tl.data.AnnDataModule(
        adata,
        genome=genome,
        batch_size=BATCH_SIZE,
        max_stochastic_shift=3,
        always_reverse_complement=True,
    )

    num_classes = adata.n_obs
    model_architecture = crested.tl.zoo.deeptopic_cnn(seq_len=SEQ_LEN, num_classes=num_classes)

    config = crested.tl.default_configs("topic_classification")
    print("config:", config)

    os.chdir(str(RUN_DIR))

    trainer = crested.tl.Crested(
        data=datamodule,
        model=model_architecture,
        config=config,
        project_name="human_topics",
        logger=None,
    )

    trainer.fit(epochs=EPOCHS)

    trained = _get_trained_model(trainer)
    if trained is not None:
        out_model = RUN_DIR / "final_model.keras"
        try:
            trained.save(str(out_model))
            print("[OK] saved final model:", out_model)
        except Exception as e:
            print("[WARN] could not save final model via .save():", repr(e))
            print("       (checkpoints should still exist under:", RUN_DIR, ")")


if __name__ == "__main__":
    main()