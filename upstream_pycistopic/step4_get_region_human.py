from __future__ import annotations

import pickle
from pathlib import Path

BASE = Path("/cfs/klemming/projects/supr/snic2022-23-547/xinyu")
OUT = BASE / "runs" / "out"

# ---- inputs
# full peaks
IN_FULL_PKL = OUT / "1_cobj_human.pkl"

# top60k
IN_TOP60K_MAP = OUT / "region_map_top60000_1_cobj_human.tsv"

# ---- outputs
OUT_FULL_BED = OUT / "regions_human.bed"
OUT_TOP60K_BED = OUT / "regions_human60k.bed"


def parse_region_name(s: str):
    """'chr1:123-456' -> ('chr1', 123, 456)"""
    s = s.strip()
    chrom, rest = s.split(":")
    start, end = rest.split("-")
    return chrom, int(start), int(end)


def write_bed(region_names: list[str], out_bed: Path):
    items = []
    seen = set()

    for r in region_names:
        try:
            chrom, start, end = parse_region_name(r)
        except Exception:
            raise RuntimeError(f"Bad region name format: {r!r} (expected chr:start-end)")

        key = (chrom, start, end)
        if key in seen:
            continue
        seen.add(key)
        items.append(key)

    items.sort(key=lambda x: (x[0], x[1], x[2]))

    with open(out_bed, "w") as f:
        for chrom, start, end in items:
            f.write(f"{chrom}\t{start}\t{end}\n")

    print(f"[write] {out_bed} (n={len(items)})")


def load_full_regions_from_pkl(pkl_path: Path) -> list[str]:
    with open(pkl_path, "rb") as f:
        cobj = pickle.load(f)

    if not hasattr(cobj, "region_names") or cobj.region_names is None:
        raise RuntimeError(f"{pkl_path} has no cobj.region_names")

    region_names = list(cobj.region_names)
    print(f"[load] full regions from {pkl_path} (n={len(region_names)})")
    return region_names


def load_top_regions_from_map(map_path: Path) -> list[str]:
    region_names = []

    with open(map_path, "r") as f:
        header = next(f).rstrip("\n")
        if "orig_region_name" not in header:
            raise RuntimeError(f"Unexpected header in map file: {header}")

        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            _new_id, orig = line.split("\t", 1)
            region_names.append(orig)

    print(f"[load] top regions from {map_path} (n={len(region_names)})")
    return region_names


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # full
    full_regions = load_full_regions_from_pkl(IN_FULL_PKL)
    write_bed(full_regions, OUT_FULL_BED)

    # top60k
    top60k_regions = load_top_regions_from_map(IN_TOP60K_MAP)
    write_bed(top60k_regions, OUT_TOP60K_BED)

    print("[done] generated human region files")


if __name__ == "__main__":
    main()