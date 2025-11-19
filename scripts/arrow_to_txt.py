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


def write_split(ds_split, src_lang, tgt_lang, out_src, out_tgt):
    os.makedirs(os.path.dirname(out_src), exist_ok=True)
    with open(out_src, "w", encoding="utf-8") as fs, open(
        out_tgt, "w", encoding="utf-8"
    ) as ft:
        for item in ds_split:
            trans = item.get("translation") or {}
            src = (trans.get(src_lang) or "").strip()
            tgt = (trans.get(tgt_lang) or "").strip()
            if src and tgt:
                fs.write(src + "\n")
                ft.write(tgt + "\n")


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
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    for src, tgt in LANG_PAIRS:
        config = f"{src}-{tgt}"
        print(f"Processing {config}...")
        ds = load_dataset("wmt/wmt14", config, cache_dir=args.cache_dir)

        pair_dir = os.path.join(args.save_dir, config)
        os.makedirs(pair_dir, exist_ok=True)

        write_split(
            ds["train"],
            src,
            tgt,
            os.path.join(pair_dir, f"train.{src}"),
            os.path.join(pair_dir, f"train.{tgt}"),
        )
        write_split(
            ds["validation"],
            src,
            tgt,
            os.path.join(pair_dir, f"valid.{src}"),
            os.path.join(pair_dir, f"valid.{tgt}"),
        )
        write_split(
            ds["test"],
            src,
            tgt,
            os.path.join(pair_dir, f"test.{src}"),
            os.path.join(pair_dir, f"test.{tgt}"),
        )


if __name__ == "__main__":
    main()

