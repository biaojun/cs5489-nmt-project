"""
Merge per-language BPE corpora into a single multilingual dataset.

Each source sentence is prefixed with a language tag (e.g., <de>) so that
one Seq2Seq/Transformer model can translate cs/de/fr/hi/ru → en.
"""

import argparse
import os

LANG_PAIRS = [
    ("cs", "en"),
    ("de", "en"),
    ("fr", "en"),
    ("hi", "en"),
    ("ru", "en"),
]

LANG_TAGS = {src: f"<{src}>" for src, _ in LANG_PAIRS}


def merge_split(data_dir: str, split: str, out_dir: str) -> int:
    out_src = os.path.join(out_dir, f"{split}.src")
    out_tgt = os.path.join(out_dir, f"{split}.tgt")
    total = 0
    with open(out_src, "w", encoding="utf-8") as fsrc, open(
        out_tgt, "w", encoding="utf-8"
    ) as ftgt:
        for src, tgt in LANG_PAIRS:
            pair_dir = os.path.join(data_dir, f"{src}-{tgt}")
            src_path = os.path.join(pair_dir, f"{split}.src")
            tgt_path = os.path.join(pair_dir, f"{split}.tgt")
            if not (os.path.exists(src_path) and os.path.exists(tgt_path)):
                continue
            with open(src_path, "r", encoding="utf-8") as fs, open(
                tgt_path, "r", encoding="utf-8"
            ) as ft:
                for s_line, t_line in zip(fs, ft):
                    s_line = s_line.strip()
                    t_line = t_line.strip()
                    if not s_line or not t_line:
                        continue
                    prefix = LANG_TAGS[src]
                    fsrc.write(f"{prefix} {s_line}\n")
                    ftgt.write(t_line + "\n")
                    total += 1
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/root/autodl-tmp/processed_wmt14",
        help="directory containing {lang-pair}/{split}.src/.tgt files",
    )
    parser.add_argument(
        "--out-name",
        type=str,
        default="multilingual",
        help="subdirectory name for merged data",
    )
    args = parser.parse_args()

    out_dir = os.path.join(args.data_dir, args.out_name)
    os.makedirs(out_dir, exist_ok=True)

    for split in ["train", "valid", "test"]:
        count = merge_split(args.data_dir, split, out_dir)
        print(f"Merged {split}: {count} sentence pairs into {out_dir}")


if __name__ == "__main__":
    main()

