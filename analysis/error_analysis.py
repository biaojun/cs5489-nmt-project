"""
Sample bad cases for qualitative error analysis.

Given parallel test set and system outputs from SMT / Seq2Seq / Transformer,
sample N examples and dump them to a TSV file for manual labeling.
"""

import argparse
import csv
import random


def read_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, required=True)
    parser.add_argument("--ref", type=str, required=True)
    parser.add_argument("--smt", type=str, required=True)
    parser.add_argument("--seq2seq", type=str, required=True)
    parser.add_argument("--transformer", type=str, required=True)
    parser.add_argument("--num", type=int, default=10)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    src = read_lines(args.src)
    ref = read_lines(args.ref)
    smt = read_lines(args.smt)
    s2s = read_lines(args.seq2seq)
    tfm = read_lines(args.transformer)

    n = min(len(src), len(ref), len(smt), len(s2s), len(tfm))
    idxs = random.sample(range(n), min(args.num, n))

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            ["idx", "source", "reference", "smt", "seq2seq", "transformer", "error_type"]
        )
        for i in idxs:
            writer.writerow([i, src[i], ref[i], smt[i], s2s[i], tfm[i], ""])

    print(f"Wrote {len(idxs)} examples to {args.out}")


if __name__ == "__main__":
    main()

