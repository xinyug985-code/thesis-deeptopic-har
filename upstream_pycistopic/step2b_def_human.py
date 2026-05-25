from __future__ import annotations

import pickle
from pathlib import Path
from pycisTopic.lda_models import evaluate_models


BASE = Path("/cfs/klemming/projects/supr/snic2022-23-547/xinyu")
OUT = BASE / "runs" / "out"

DATASETS = {
    "human": OUT / "2_cobj_tmp_mallet_human_k40-100",
}


def load_models(model_dir):

    models = []

    for p in sorted(model_dir.glob("Topic*.pkl")):
        print("loading", p.name)
        with open(p, "rb") as f:
            models.append(pickle.load(f))

    return models


def run_eval(tag, model_dir):

    models = load_models(model_dir)

    best_model = evaluate_models(
        models,
        select_model=None,      
        return_model=True,
        plot=True,              
        plot_metrics=True,      
        save=str(OUT / f"lda_eval_def_{tag}.pdf")  
    )

    print("\nBest model topic number:", best_model.n_topic)


def main():

    for tag, model_dir in DATASETS.items():
        run_eval(tag, model_dir)


if __name__ == "__main__":
    main()