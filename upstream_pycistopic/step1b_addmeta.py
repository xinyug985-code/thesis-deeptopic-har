from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

BASE = Path("/cfs/klemming/projects/supr/snic2022-23-547/xinyu")
OUT = BASE / "runs" / "out"

DATASETS = {
    "human": {
        "cobj_pkl": OUT / "1_cobj_human.pkl",
        "meta_tsv": BASE / "metadata_pycistopic.tsv",
        "report_tsv": OUT / "step1_validbc_report_human.tsv",
        "id_col": "cell_id",
        "class_col": "final_class",
        "keep_meta_cols": ["cell_id", "cell_type", "Sample", "final_class"],
        "out_pkl": OUT / "1_cobj_human.withmeta.pkl",
    },
    "macaque": {
        "cobj_pkl": OUT / "1_cobj_macaque.pkl",
        "meta_tsv": BASE / "metadata_pycistopic_macaque.tsv",
        "report_tsv": OUT / "step1_validbc_report_macaque.tsv",
        "id_col": "Unnamed: 0",
        "class_col": "annotation",
        "keep_meta_cols": ["Unnamed: 0", "sample_id", "annotation"],
        "out_pkl": OUT / "1_cobj_macaque.withmeta.pkl",
    },
}


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(obj, path: Path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def filter_unknown(meta: pd.DataFrame, class_col: str) -> pd.DataFrame:
    cls = meta[class_col].fillna("unknown").astype("string").str.strip().str.lower()
    unknown_set = {"unknown", "na", "nan", ""}
    return meta.loc[~cls.isin(unknown_set)].copy()


def parse_meta_cell_id(cell_id: str, sid_to_delim: dict[str, str]) -> tuple[str | None, str | None]:
    """
    Parse metadata full cell_id into (sample_id, barcode)
    using the exact sample_id -> delimiter map from step1 report.
    """
    if pd.isna(cell_id):
        return None, None

    cell_id = str(cell_id)

    # longest sample_id first, avoid partial prefix mismatch
    for sid in sorted(sid_to_delim.keys(), key=len, reverse=True):
        delim = sid_to_delim[sid]
        prefix = f"{sid}{delim}"
        if cell_id.startswith(prefix):
            barcode = cell_id[len(prefix):]
            return sid, barcode

    return None, None


def strip_cobj_index_suffix(idx: pd.Index) -> pd.Series:
    """
    cobj merged index often becomes:
      original_cell_id + '-xinyu_topicpipe_...___...'
    We don't use it for merging, but keep a cleaned version for debugging.
    """
    s = idx.astype(str)
    s = s.str.replace(r"-xinyu_topicpipe.*$", "", regex=True)
    return s


def add_metadata_one(tag: str, cfg: dict):
    print(f"\n===== [{tag}] =====")
    print("[load] cobj :", cfg["cobj_pkl"])
    print("[load] meta :", cfg["meta_tsv"])
    print("[load] report:", cfg["report_tsv"])

    cobj = load_pickle(cfg["cobj_pkl"])
    meta = pd.read_csv(cfg["meta_tsv"], sep="\t", dtype="string")
    report = pd.read_csv(cfg["report_tsv"], sep="\t", dtype="string")

    print("[cobj] n_cells =", cobj.cell_data.shape[0])
    print("[meta] total rows =", meta.shape[0])

    meta = filter_unknown(meta, cfg["class_col"])
    print(f"[meta] after filtering {cfg['class_col']} Unknown =", meta.shape[0])

    sid_to_delim = dict(zip(report["sample_id"].astype(str), report["delimiter"].astype(str)))

    keep_cols = [c for c in cfg["keep_meta_cols"] if c in meta.columns]
    if cfg["id_col"] not in keep_cols:
        keep_cols = [cfg["id_col"]] + keep_cols
    meta = meta[keep_cols].copy()

    # parse metadata side
    parsed_meta = meta[cfg["id_col"]].apply(lambda x: parse_meta_cell_id(x, sid_to_delim))
    meta["sample_id_parsed"] = parsed_meta.str[0]
    meta["barcode_parsed"] = parsed_meta.str[1]

    print("[meta] parsed sample_id non-NA =", int(meta["sample_id_parsed"].notna().sum()), "/", meta.shape[0])

    # parse cobj side FROM CLEANED INDEX, not from cobj.sample_id / barcode
    cell_df = cobj.cell_data.copy()
    cell_df["original_cell_id_clean"] = strip_cobj_index_suffix(cell_df.index)

    parsed_cobj = cell_df["original_cell_id_clean"].apply(lambda x: parse_meta_cell_id(x, sid_to_delim))
    cell_df["sample_id_parsed"] = parsed_cobj.str[0]
    cell_df["barcode_parsed"] = parsed_cobj.str[1]

    print("\n[cobj parsed head]")
    print(cell_df[["original_cell_id_clean", "sample_id_parsed", "barcode_parsed"]].head())

    print("\n[meta parsed head]")
    print(meta[[cfg["id_col"], "sample_id_parsed", "barcode_parsed"]].head())

    meta_sub = meta.drop_duplicates(subset=["sample_id_parsed", "barcode_parsed"])

    merged = cell_df.merge(
        meta_sub,
        on=["sample_id_parsed", "barcode_parsed"],
        how="left",
        suffixes=("", "_meta"),
    )
    merged.index = cobj.cell_data.index
    cobj.cell_data = merged

    n_matched = int(cobj.cell_data[cfg["class_col"]].notna().sum()) if cfg["class_col"] in cobj.cell_data.columns else 0
    print(f"[merge] matched cells by {cfg['class_col']} =", n_matched, "/", cobj.cell_data.shape[0])

    print("[cell_data cols] =", list(cobj.cell_data.columns))

    for col in cfg["keep_meta_cols"]:
        if col in cobj.cell_data.columns:
            print(f"\n[value counts] {col}")
            print(cobj.cell_data[col].value_counts(dropna=False).head(20))

    save_pickle(cobj, cfg["out_pkl"])
    print("[save] wrote:", cfg["out_pkl"])


def main():
    for tag, cfg in DATASETS.items():
        add_metadata_one(tag, cfg)


if __name__ == "__main__":
    main()