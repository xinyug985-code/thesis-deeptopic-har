from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import List

import numpy as np
from pycisTopic.lda_models import run_cgs_models_mallet

BASE = Path("/cfs/klemming/projects/supr/snic2022-23-547/xinyu")
OUT = BASE / "runs" / "out"

IN_PKL = OUT / "1_cobj_macaque.pkl"

TOPICS: List[int] = [40, 50, 60, 70, 80, 90, 100]

N_ITER = 500
N_CPU = int(os.environ.get("SLURM_CPUS_PER_TASK", "32"))
RANDOM_STATE = 0

MALLET = "/cfs/klemming/projects/supr/snic2022-23-547/xinyu/tools/mallet-2.0.8/bin/mallet_mem"

TAG = "macaque_k40-100"
TMP_DIR = OUT / f"2_cobj_tmp_mallet_{TAG}"
OUT_PKL = OUT / f"2_cobj_lda_mallet_{TAG}.pkl"


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(obj, path: Path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def rebuild_binary_matrix(cobj):
    X = cobj.fragment_matrix
    Xb = X.copy()
    Xb.data = np.ones_like(Xb.data)
    cobj.binary_matrix = Xb.tocsr()
    return cobj


def main():
    if not IN_PKL.exists():
        raise FileNotFoundError(f"Missing input: {IN_PKL}")
    if not os.path.exists(MALLET):
        raise FileNotFoundError(f"Missing MALLET: {MALLET}")

    TMP_DIR.mkdir(parents=True, exist_ok=True)

    cobj = load_pickle(IN_PKL)
    print("[load] ok:", IN_PKL)

    print("[matrix] fragment_matrix shape=", cobj.fragment_matrix.shape, "nnz=", cobj.fragment_matrix.nnz)
    print("[names] n_cells=", len(cobj.cell_names), "n_regions=", len(cobj.region_names))

    cobj = rebuild_binary_matrix(cobj)
    Xb = cobj.binary_matrix.tocsc()
    print(
        "[sanity] binary_matrix shape=", Xb.shape,
        "max_row_index=", int(Xb.indices.max()) if Xb.nnz else -1
    )

    for k in TOPICS:
        out_topic = TMP_DIR / f"Topic{k}.pkl"
        if out_topic.exists():
            print(f"[skip] Topic{k} exists: {out_topic}")
            continue

        print(f"[run] k={k} iter={N_ITER} cpu={N_CPU}")
        _models = run_cgs_models_mallet(
            cistopic_obj=cobj,
            n_topics=[k],
            n_cpu=N_CPU,
            n_iter=N_ITER,
            random_state=RANDOM_STATE,
            tmp_path=str(TMP_DIR),
            save_path=str(TMP_DIR),
            reuse_corpus=True,
            mallet_path=MALLET,
        )
        print(f"[done] k={k} -> {out_topic}")

    models_loaded = []
    for k in TOPICS:
        p = TMP_DIR / f"Topic{k}.pkl"
        if p.exists():
            with open(p, "rb") as f:
                models_loaded.append(pickle.load(f))

    for m in models_loaded:
        try:
            cobj.add_LDA_model(m)
        except Exception:
            pass

    save_pickle(cobj, OUT_PKL)
    print("[save] wrote:", OUT_PKL)
    print("[tmp] models in:", TMP_DIR)


if __name__ == "__main__":
    main()