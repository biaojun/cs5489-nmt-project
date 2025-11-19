"""
Clean parallel corpora:
- remove empty lines
- remove too short / too long sentences
- remove pairs with large length ratio
- strip simple HTML/XML tags

Input:  train/valid/test .{src,.tgt} (per language pair)
Output: train/valid/test .clean.{src,.tgt}
"""

import argparse
import os
import re
from typing import Iterable, Tuple


HTML_TAG_RE = re.compile(r"<[^>]+>")


def clean_pair(
    src: str,
    tgt: str,
    min_len: int,
    max_len: int,
    max_len_ratio: float,
) -> Tuple[bool, str, str]:
    src = src.strip()
    tgt = tgt.strip()
    if not src or not tgt:
        return False, "", ""

    # strip simple HTML/XML tags
    src_stripped = HTML_TAG_RE.sub("", src)
    tgt_stripped = HTML_TAG_RE.sub("", tgt)
    src_stripped = src_stripped.strip()
    tgt_stripped = tgt_stripped.strip()
    if not src_stripped or not tgt_stripped:
        return False, "", ""

    src_tokens = src_stripped.split()
    tgt_tokens = tgt_stripped.split()

    if len(src_tokens) < min_len or len(tgt_tokens) < min_len:
        return False, "", ""
    if len(src_tokens) > max_len or len(tgt_tokens) > max_len:
        return False, "", ""

    ratio = len(src_tokens) / max(len(tgt_tokens), 1)
    if ratio > max_len_ratio or ratio < 1.0 / max_len_ratio:
        return False, "", ""

    return True, " ".join(src_tokens), " ".join(tgt_tokens)


def process_file_pair(
    src_path: str,
    tgt_path: str,
    out_src_path: str,
    out_tgt_path: str,
    min_len: int,
    max_len: int,
    max_len_ratio: float,
) -> Tuple[int, int]:
    os.makedirs(os.path.dirname(out_src_path), exist_ok=True)
    kept, total = 0, 0
    with open(src_path, "r", encoding="utf-8") as fs, open(
        tgt_path, "r", encoding="utf-8"
    ) as ft, open(out_src_path, "w", encoding="utf-8") as fos, open(
        out_tgt_path, "w", encoding="utf-8"
    ) as fot:
        for src, tgt in zip(fs, ft):
            total += 1
            ok, src_c, tgt_c = clean_pair(src, tgt, min_len, max_len, max_len_ratio)
            if ok:
                fos.write(src_c + "\n")
                fot.write(tgt_c + "\n")
                kept += 1
    return kept, total


def iter_lang_pairs(root: str) -> Iterable[Tuple[str, str]]:
    for name in os.listdir(root):
        if "-" in name:
            yield name, os.path.join(root, name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/root/autodl-tmp/processed_wmt14",
        help="directory with {lang-pair}/train.src,train.tgt,... files",
    )
    parser.add_argument("--min-len", type=int, default=3)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--max-len-ratio", type=float, default=3.0)
    args = parser.parse_args()

    for pair_name, pair_dir in iter_lang_pairs(args.data_dir):
        src, tgt = pair_name.split("-")
        print(f"Cleaning {pair_name}...")
        for split in ["train", "valid", "test"]:
            in_src = os.path.join(pair_dir, f"{split}.{src}")
            in_tgt = os.path.join(pair_dir, f"{split}.{tgt}")
            out_src = os.path.join(pair_dir, f"{split}.clean.{src}")
            out_tgt = os.path.join(pair_dir, f"{split}.clean.{tgt}")
            if not (os.path.exists(in_src) and os.path.exists(in_tgt)):
                print(f"  Skip {split}, files not found")
                continue
            kept, total = process_file_pair(
                in_src,
                in_tgt,
                out_src,
                out_tgt,
                args.min_len,
                args.max_len,
                args.max_len_ratio,
            )
            print(f"  {split}: kept {kept}/{total} sentence pairs")


if __name__ == "__main__":
    main()

