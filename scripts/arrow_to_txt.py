#!/usr/bin/env python3
"""
Convert WMT14 Arrow datasets (HuggingFace `wmt/wmt14`) to plain text
parallel corpora for MT training.

This script reads local Arrow shards (no HF download) and writes
train/valid/test files for cs/de/fr/hi/ru → en.
"""

import argparse
import glob
import os

from datasets import load_dataset


LANG_PAIRS = [
    ("cs", "en"),
    ("de", "en"),
    ("fr", "en"),
    ("hi", "en"),
    ("ru", "en"),
]


def write_split(ds_split, src_lang, tgt_lang, out_src, out_tgt, max_lines=None):
    os.makedirs(os.path.dirname(out_src), exist_ok=True)
    n_written = 0
    with open(out_src, "w", encoding="utf-8") as fsrc, open(
        out_tgt, "w", encoding="utf-8"
    ) as ftgt:
        for item in ds_split:
            if max_lines is not None and n_written >= max_lines:
                break
            trans = item.get("translation") or {}
            src = (trans.get(src_lang) or "").strip()
            tgt = (trans.get(tgt_lang) or "").strip()
            if src and tgt:
                fsrc.write(src + "\n")
                ftgt.write(tgt + "\n")
                n_written += 1
    print(
        f"  Wrote {n_written} sentence pairs to "
        f"{os.path.basename(out_src)} / {os.path.basename(out_tgt)}"
    )
    return n_written


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="/root/autodl-tmp/data/wmt___wmt14",
        help="directory containing {lang-pair}/wmt14-*.arrow files",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="/root/autodl-tmp/processed_wmt14",
        help="output directory for plain text files",
    )
    parser.add_argument(
        "--mode",
        choices=["full", "subset"],
        default="full",
        help="use all training data or only a subset",
    )
    parser.add_argument(
        "--max-train-lines",
        type=int,
        default=100000,
        help="when mode=subset, cap the number of training sentence pairs per language",
    )
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    for src, tgt in LANG_PAIRS:
        config = f"{src}-{tgt}"
        print(f"Processing {config}...")
        pair_dir = os.path.join(args.cache_dir, config)
        train_pattern = os.path.join(pair_dir, "wmt14-train*.arrow")
        train_files = sorted(glob.glob(train_pattern))
        if not train_files:
            raise FileNotFoundError(f"No train Arrow files matching {train_pattern}")
        data_files = {
            "train": train_files,
            "validation": os.path.join(pair_dir, "wmt14-validation.arrow"),
            "test": os.path.join(pair_dir, "wmt14-test.arrow"),
        }
        ds = load_dataset("arrow", data_files=data_files)

        out_dir = os.path.join(args.save_dir, config)
        os.makedirs(out_dir, exist_ok=True)

        max_train = None
        if args.mode == "subset" and args.max_train_lines > 0:
            max_train = args.max_train_lines

        n_train = write_split(
            ds["train"],
            src,
            tgt,
            os.path.join(out_dir, f"train.{src}"),
            os.path.join(out_dir, f"train.{tgt}"),
            max_lines=max_train,
        )
        n_valid = write_split(
            ds["validation"],
            src,
            tgt,
            os.path.join(out_dir, f"valid.{src}"),
            os.path.join(out_dir, f"valid.{tgt}"),
        )
        n_test = write_split(
            ds["test"],
            src,
            tgt,
            os.path.join(out_dir, f"test.{src}"),
            os.path.join(out_dir, f"test.{tgt}"),
        )
        print(
            f"  Summary for {config}: train={n_train}, valid={n_valid}, test={n_test}"
        )


if __name__ == "__main__":
    main()

