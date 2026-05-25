from __future__ import annotations

import pickle
from pathlib import Path

from pycisTopic.topic_binarization import binarize_topics
from pycisTopic.utils import region_names_to_coordinates


# =====================
# Config
# =====================
BASE = Path("/cfs/klemming/projects/supr/snic2022-23-547/xinyu")
OUT = BASE / "runs" / "out"

IN_COBJ = OUT / "1_cobj_macaque.pkl"

IN_MODEL = OUT / "2_cobj_tmp_mallet_macaque_k40-100" / "Topic100.pkl"

BEDDIR = OUT / "topic_beds_macaque_otsu"
BEDDIR.mkdir(parents=True, exist_ok=True)

OUT_BIN_PKL = OUT / "3_topic_binarized_macaque_otsu.pkl"

METHOD = "otsu"

# if METHOD = "ntop"
NTOP = 3000


# =====================
# Helpers
# =====================
def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    if not IN_COBJ.exists():
        raise FileNotFoundError(f"Missing cisTopic object: {IN_COBJ}")
    if not IN_MODEL.exists():
        raise FileNotFoundError(f"Missing model: {IN_MODEL}")

    print("[load] cistopic_obj:", IN_COBJ)
    cistopic_obj = load_pickle(IN_COBJ)

    print("[load] model:", IN_MODEL)
    model = load_pickle(IN_MODEL)

    # binarize_topics() 
    cistopic_obj.selected_model = model
    print(f"[model] selected_model.n_topic = {model.n_topic}")

    # binarization
    # Otsu / Yen ~ topic-region；ntop ~ region set
    print(f"[binarize] method = {METHOD}")
    if METHOD == "ntop":
        region_bin_topics = binarize_topics(
            cistopic_obj,
            target="region",
            method="ntop",
            ntop=NTOP,
            plot=True,
            num_columns=5,
        )
    else:
        region_bin_topics = binarize_topics(
            cistopic_obj,
            target="region",
            method=METHOD,
            plot=True,
            num_columns=5,
        )

    with open(OUT_BIN_PKL, "wb") as f:
        pickle.dump(region_bin_topics, f)
    print("[save] binarized topics ->", OUT_BIN_PKL)

    n_topics = 0
    for topic in region_bin_topics:
        df = region_names_to_coordinates(region_bin_topics[topic].index)
        df = df.sort_values(["Chromosome", "Start", "End"])

        out_bed = BEDDIR / f"{topic}.bed"
        df.to_csv(out_bed, sep="\t", header=False, index=False)

        print(f"[export] {topic}: {df.shape[0]} peaks -> {out_bed.name}")
        n_topics += 1

    print(f"[done] wrote {n_topics} BED files to {BEDDIR}")


if __name__ == "__main__":
    main()