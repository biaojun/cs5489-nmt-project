"""
Train a joint SentencePiece BPE model on all cleaned training data.

Input:  {lang-pair}/train.clean.{src,tgt}
Output: spm/spm.model, spm/spm.vocab
"""

import argparse
import os
import tempfile

import sentencepiece as spm


LANG_PAIRS = [
    ("cs", "en"),
    ("de", "en"),
    ("fr", "en"),
    ("hi", "en"),
    ("ru", "en"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/root/autodl-tmp/processed_wmt14",
        help="directory with cleaned train.clean.* files",
    )
    parser.add_argument(
        "--model-prefix",
        type=str,
        default="spm",
        help="SentencePiece model prefix (under --data-dir/spm)",
    )
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--model-type", type=str, default="bpe")
    args = parser.parse_args()

    spm_dir = os.path.join(args.data_dir, "spm")
    os.makedirs(spm_dir, exist_ok=True)
    model_prefix = os.path.join(spm_dir, args.model_prefix)

    with tempfile.NamedTemporaryFile(mode="w+", encoding="utf-8", delete=False) as tmp:
        tmp_path = tmp.name
        for src, tgt in LANG_PAIRS:
            pair_dir = os.path.join(args.data_dir, f"{src}-{tgt}")
            src_path = os.path.join(pair_dir, f"train.clean.{src}")
            tgt_path = os.path.join(pair_dir, f"train.clean.{tgt}")
            if not (os.path.exists(src_path) and os.path.exists(tgt_path)):
                continue
            for path in [src_path, tgt_path]:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            tmp.write(line + "\n")

    spm.SentencePieceTrainer.Train(
        input=tmp_path,
        model_prefix=model_prefix,
        vocab_size=args.vocab_size,
        model_type=args.model_type,
        character_coverage=1.0,
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
        input_sentence_size=10000000,
        shuffle_input_sentence=True,
    )
    os.remove(tmp_path)
    print(f"Trained SentencePiece model at {model_prefix}.model")


if __name__ == "__main__":
    main()

