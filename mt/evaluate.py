import argparse
import json
import os

import sacrebleu
import sentencepiece as spm
import torch
from nltk.translate.meteor_score import meteor_score

from mt.data import PAD_ID, make_dataloader
from mt.models import Seq2SeqModel, Seq2SeqConfig, TransformerModel, TransformerConfig


def load_model(path: str, device: str):
    ckpt = torch.load(path, map_location=device)
    model_type = ckpt["model_type"]
    config_dict = ckpt["config"]
    if model_type == "seq2seq":
        config = Seq2SeqConfig(**config_dict)
        model = Seq2SeqModel(config)
    else:
        config = TransformerConfig(**config_dict)
        model = TransformerModel(config)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, ckpt["spm_model"], model_type


def decode_ids(sp, ids):
    tokens = [sp.id_to_piece(int(t)) for t in ids if int(t) not in (PAD_ID, 1, 2)]
    text = "".join(tokens).replace("▁", " ").strip()
    return text


def evaluate_model(
    ckpt_path: str,
    data_dir: str,
    lang_pair: str,
    split: str,
    device: str,
    out_path: str,
):
    model, spm_model, model_type = load_model(ckpt_path, device)
    src, tgt = lang_pair.split("-")
    pair_dir = os.path.join(data_dir, lang_pair)
    src_path = os.path.join(pair_dir, f"{split}.src")
    tgt_path = os.path.join(pair_dir, f"{split}.tgt")

    sp = spm.SentencePieceProcessor(model_file=spm_model)
    loader = make_dataloader(src_path, tgt_path, spm_model, batch_size=64, shuffle=False)

    all_refs = []
    all_hyps = []
    all_src = []
    with torch.no_grad():
        for batch in loader:
            src_batch = batch.src.to(device)
            src_lens = batch.src_lens.to(device)
            if model_type == "seq2seq":
                hyps_ids = model.greedy_decode(src_batch, src_lens)
            else:
                hyps_ids = model.greedy_decode(src_batch)
            for i in range(src_batch.size(0)):
                hyp = decode_ids(sp, hyps_ids[i])
                ref = decode_ids(sp, batch.tgt_output[i])
                src_text = decode_ids(sp, batch.src[i])
                all_hyps.append(hyp)
                all_refs.append(ref)
                all_src.append(src_text)

    bleu = sacrebleu.corpus_bleu(all_hyps, [all_refs])

    meteor_scores = []
    for hyp, ref in zip(all_hyps, all_refs):
        meteor_scores.append(meteor_score([ref.split()], hyp.split()))
    meteor = sum(meteor_scores) / max(len(meteor_scores), 1)

    metrics = {
        "model": model_type,
        "lang_pair": lang_pair,
        "split": split,
        "bleu": float(bleu.score),
        "meteor": float(meteor),
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))

    # Save plain-text outputs for error analysis
    out_txt = out_path.replace(".json", ".txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        for s, r, h in zip(all_src, all_refs, all_hyps):
            f.write(f"SRC\t{s}\nREF\t{r}\nHYP\t{h}\n\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/root/autodl-tmp/processed_wmt14",
    )
    parser.add_argument("--lang-pair", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    evaluate_model(
        ckpt_path=args.ckpt,
        data_dir=args.data_dir,
        lang_pair=args.lang_pair,
        split=args.split,
        device=args.device,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()

