"""
Compute basic statistics for each language pair:
- number of sentence pairs
- average source/target length (tokens)

Runs on preprocessed BPE corpora: {lang-pair}/{split}.src/.tgt
"""

import argparse
import os
from typing import Dict, Tuple


def stats_for_file(path: str) -> Tuple[int, float]:
    n = 0
    total_len = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            toks = line.strip().split()
            if not toks:
                continue
            n += 1
            total_len += len(toks)
    avg_len = total_len / max(n, 1)
    return n, avg_len


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/root/autodl-tmp/processed_wmt14",
    )
    parser.add_argument("--split", type=str, default="train")
    args = parser.parse_args()

    results = {}
    for name in os.listdir(args.data_dir):
        if "-" not in name:
            continue
        pair_dir = os.path.join(args.data_dir, name)
        src_path = os.path.join(pair_dir, f"{args.split}.src")
        tgt_path = os.path.join(pair_dir, f"{args.split}.tgt")
        if not (os.path.exists(src_path) and os.path.exists(tgt_path)):
            continue
        n, src_avg = stats_for_file(src_path)
        _, tgt_avg = stats_for_file(tgt_path)
        results[name] = {
            "n_sentences": n,
            "avg_src_len": src_avg,
            "avg_tgt_len": tgt_avg,
        }

    for pair, stats in sorted(results.items()):
        print(
            f"{pair} ({args.split}): "
            f"n={stats['n_sentences']}, "
            f"avg_src_len={stats['avg_src_len']:.1f}, "
            f"avg_tgt_len={stats['avg_tgt_len']:.1f}"
        )


if __name__ == "__main__":
    main()

