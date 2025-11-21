#!/usr/bin/env python3
"""
Apply a trained SentencePiece model to cleaned corpora and generate
model-ready .src/.tgt files for each language pair.

Input:  {lang-pair}/{split}.clean.{src,tgt}
Output: {lang-pair}/{split}.src, {lang-pair}/{split}.tgt (BPE tokenized)
"""

import argparse
import os
from typing import Iterable, Tuple

import sentencepiece as spm


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
        help="directory with cleaned corpora",
    )
    parser.add_argument(
        "--spm-model",
        type=str,
        default="/root/autodl-tmp/processed_wmt14/spm/spm.model",
        help="path to trained SentencePiece model",
    )
    args = parser.parse_args()

    sp = spm.SentencePieceProcessor(model_file=args.spm_model)

    for pair_name, pair_dir in iter_lang_pairs(args.data_dir):
        src, tgt = pair_name.split("-")
        print(f"Applying SentencePiece to {pair_name}...")
        for split in ["train", "valid", "test"]:
            in_src = os.path.join(pair_dir, f"{split}.clean.{src}")
            in_tgt = os.path.join(pair_dir, f"{split}.clean.{tgt}")
            if not (os.path.exists(in_src) and os.path.exists(in_tgt)):
                continue
            out_src = os.path.join(pair_dir, f"{split}.src")
            out_tgt = os.path.join(pair_dir, f"{split}.tgt")
            with open(in_src, "r", encoding="utf-8") as fs, open(
                in_tgt, "r", encoding="utf-8"
            ) as ft, open(out_src, "w", encoding="utf-8") as fos, open(
                out_tgt, "w", encoding="utf-8"
            ) as fot:
                for src_line, tgt_line in zip(fs, ft):
                    src_tokens = sp.encode(src_line.strip(), out_type=str)
                    tgt_tokens = sp.encode(tgt_line.strip(), out_type=str)
                    fos.write(" ".join(src_tokens) + "\n")
                    fot.write(" ".join(tgt_tokens) + "\n")


if __name__ == "__main__":
    main()

