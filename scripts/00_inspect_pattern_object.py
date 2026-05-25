#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Inspect actual structure of one saved TF-MoDISco pattern object.

Run:
    python scripts/00_inspect_pattern_object.py --species human --pattern-key 0
"""

from pathlib import Path
import argparse
import pickle
import numpy as np

BASE = Path("/mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/pjt")
STAGE2 = BASE / "runs/out/stage2_topic_modisco_top1000"


def safe_shape(x):
    try:
        return np.asarray(x).shape
    except Exception as e:
        return f"ERR: {type(e).__name__}: {e}"


def inspect_obj(name, obj, depth=0, max_depth=2):
    indent = "  " * depth
    print(f"\n{indent}=== {name} ===")
    print(f"{indent}type:", type(obj))
    print(f"{indent}repr head:", repr(obj)[:300])
    print(f"{indent}np.asarray shape:", safe_shape(obj))

    if isinstance(obj, dict):
        print(f"{indent}dict keys:", list(obj.keys()))
        if depth < max_depth:
            for k, v in obj.items():
                inspect_obj(f"{name}[{k!r}]", v, depth + 1, max_depth)
        return

    attrs = [a for a in dir(obj) if not a.startswith("_")]
    print(f"{indent}attrs:", attrs[:80])

    interesting = [
        "ppm", "pwm", "cwm", "contrib_scores", "contribs", "sequence", "seq",
        "forward_pattern", "reverse_pattern", "pattern", "seqlet_set",
        "name", "task_names", "hypothetical_contribs", "hypothetical_contrib_scores",
    ]

    for a in interesting:
        if hasattr(obj, a):
            try:
                v = getattr(obj, a)
                print(f"{indent}attr {a}: type={type(v)}, shape={safe_shape(v)}, repr={repr(v)[:120]}")
            except Exception as e:
                print(f"{indent}attr {a}: ERR {type(e).__name__}: {e}")

    # Try methods that often exist in modisco-lite objects
    methods = [
        "get_seq_ic", "get_task_contrib_scores", "get_contrib_scores",
        "get_hypothetical_contribs", "get_sequence", "trim",
    ]

    for m in methods:
        if hasattr(obj, m):
            try:
                fn = getattr(obj, m)
                print(f"{indent}method {m}: {fn}")
            except Exception as e:
                print(f"{indent}method {m}: ERR {type(e).__name__}: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--species", choices=["human", "macaque"], default="human")
    parser.add_argument("--pattern-key", default="0")
    args = parser.parse_args()

    pattern_pkl = STAGE2 / args.species / "pattern_matrix" / f"{args.species}_all_patterns.pkl"
    print("[load]", pattern_pkl)

    with open(pattern_pkl, "rb") as f:
        all_patterns = pickle.load(f)

    print("\nall_patterns type:", type(all_patterns))
    print("example keys:", list(all_patterns.keys())[:10])

    pat_outer = all_patterns[str(args.pattern_key)]
    inspect_obj("outer pattern dict", pat_outer, max_depth=2)


if __name__ == "__main__":
    main()
