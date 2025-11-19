"""
Convert WMT14 Arrow datasets (HuggingFace `wmt/wmt14`) to plain text
parallel corpora for traditional MT.

This follows the reference code snippet in the assignment prompt.
It assumes that Arrow files are under `/root/autodl-tmp/data/wmt___wmt14/`
in the environment where you run it, but you can override paths via CLI.
"""

import argparse
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
    """
    Write one split to plain-text files and return number of sentence pairs.

    If max_lines is not None, at most that many valid sentence pairs are written.
    """
    os.makedirs(os.path.dirname(out_src), exist_ok=True)
    n_written = 0
    with open(out_src, "w", encoding="utf-8") as fs, open(
        out_tgt, "w", encoding="utf-8"
    ) as ft:
        for item in ds_split:
            if max_lines is not None and n_written >= max_lines:
                break
            trans = item.get("translation") or {}
            src = (trans.get(src_lang) or "").strip()
            tgt = (trans.get(tgt_lang) or "").strip()
            if src and tgt:
                fs.write(src + "\n")
                ft.write(tgt + "\n")
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
        default="/root/autodl-tmp/data",
        help="datasets cache directory (where wmt___wmt14 lives)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="/root/autodl-tmp/processed_wmt14",
        help="where to save plain text files",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["full", "subset"],
        default="full",
        help="use all training data or only a subset",
    )
    parser.add_argument(
        "--max-train-lines",
        type=int,
        default=100000,
        help=(
            "when mode=subset, maximum number of training sentence pairs per language "
            "pair (set to 0 to disable limit)"
        ),
    )
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    for src, tgt in LANG_PAIRS:
        config = f"{src}-{tgt}"
        print(f"Processing {config}...")

        # In this project we work fully离线：Arrow 文件已经在本地，
        # 所以直接用内置的 `arrow` 数据集读取本地 .arrow 文件，
        # 不再访问 HuggingFace Hub。
        pair_cache_dir = os.path.join(args.cache_dir, config)
        data_files = {
            "train": os.path.join(pair_cache_dir, "wmt14-train*.arrow"),
            "validation": os.path.join(pair_cache_dir, "wmt14-validation.arrow"),
            "test": os.path.join(pair_cache_dir, "wmt14-test.arrow"),
        }
        ds = load_dataset("arrow", data_files=data_files)

        pair_dir = os.path.join(args.save_dir, config)
        os.makedirs(pair_dir, exist_ok=True)

        max_train = None
        if args.mode == "subset" and args.max_train_lines > 0:
            max_train = args.max_train_lines

        n_train = write_split(
            ds["train"],
            src,
            tgt,
            os.path.join(pair_dir, f"train.{src}"),
            os.path.join(pair_dir, f"train.{tgt}"),
            max_lines=max_train,
        )
        n_valid = write_split(
            ds["validation"],
            src,
            tgt,
            os.path.join(pair_dir, f"valid.{src}"),
            os.path.join(pair_dir, f"valid.{tgt}"),
        )
        n_test = write_split(
            ds["test"],
            src,
            tgt,
            os.path.join(pair_dir, f"test.{src}"),
            os.path.join(pair_dir, f"test.{tgt}"),
        )
        print(
            f"  Summary for {config}: "
            f"train={n_train}, valid={n_valid}, test={n_test}"
        )


if __name__ == "__main__":
    main()
