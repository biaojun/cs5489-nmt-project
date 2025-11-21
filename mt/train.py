import argparse
import math
import os
from typing import Tuple

import sacrebleu
import sentencepiece as spm
import torch
import torch.nn as nn
from torch.optim import Adam

from mt.data import PAD_ID, make_dataloader
from mt.models import Seq2SeqModel, Seq2SeqConfig, TransformerModel, TransformerConfig


def label_smoothed_nll_loss(logits, target, epsilon, ignore_index=PAD_ID):
    lprobs = torch.log_softmax(logits, dim=-1)
    nll_loss = nn.functional.nll_loss(
        lprobs.view(-1, lprobs.size(-1)),
        target.view(-1),
        ignore_index=ignore_index,
        reduction="sum",
    )
    smooth_loss = -lprobs.sum(dim=-1).view(-1)
    non_pad_mask = target.view(-1) != ignore_index
    smooth_loss = smooth_loss[non_pad_mask].sum()
    n_tokens = non_pad_mask.sum()
    eps_i = epsilon / logits.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss / n_tokens


def build_model(model_type: str, vocab_size: int):
    if model_type == "seq2seq":
        config = Seq2SeqConfig(vocab_size=vocab_size)
        return Seq2SeqModel(config)
    elif model_type == "transformer":
        config = TransformerConfig(vocab_size=vocab_size)
        return TransformerModel(config)
    else:
        raise ValueError(f"Unknown model_type {model_type}")


def build_optim(model, model_type: str, d_model: int = 512, warmup: int = 4000):
    if model_type == "seq2seq":
        return Adam(model.parameters(), lr=1e-3), None

    optimizer = Adam(
        model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9
    )

    def lr_lambda(step):
        if step == 0:
            step = 1
        return (d_model ** -0.5) * min(step ** -0.5, step * (warmup ** -1.5))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler


def compute_bleu(sp, model, loader, device) -> float:
    model.eval()
    all_refs = []
    all_hyps = []
    with torch.no_grad():
        for batch in loader:
            src = batch.src.to(device)
            src_lens = batch.src_lens.to(device)
            if hasattr(model, "greedy_decode"):
                if isinstance(model, Seq2SeqModel):
                    hyps_ids = model.greedy_decode(src, src_lens)
                else:
                    hyps_ids = model.greedy_decode(src)
            else:
                continue
            for hyp_ids, tgt_ids in zip(hyps_ids, batch.tgt_output):
                hyp_tokens = [
                    sp.id_to_piece(int(t))
                    for t in hyp_ids
                    if int(t) not in (PAD_ID, 1, 2)
                ]
                ref_tokens = [
                    sp.id_to_piece(int(t))
                    for t in tgt_ids
                    if int(t) not in (PAD_ID,)
                ]
                all_hyps.append("".join(hyp_tokens).replace("▁", " ").strip())
                all_refs.append("".join(ref_tokens).replace("▁", " ").strip())
    bleu = sacrebleu.corpus_bleu(all_hyps, [all_refs])
    model.train()
    return float(bleu.score)


def run_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    device,
    epoch_idx: int,
    num_epochs: int,
    model_type: str,
) -> float:
    model.train()
    total_loss = 0.0
    total_tokens = 0
    ce_loss = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    if model_type == "seq2seq":
        start_tf = 1.0
        end_tf = 0.5
        tf_ratio = start_tf - (epoch_idx) * (start_tf - end_tf) / max(num_epochs - 1, 1)
    else:
        tf_ratio = 1.0

    for step, batch in enumerate(loader, start=1):
        batch = batch
        batch.src = batch.src.to(device)
        batch.src_lens = batch.src_lens.to(device)
        batch.tgt_input = batch.tgt_input.to(device)
        batch.tgt_output = batch.tgt_output.to(device)

        optimizer.zero_grad()
        if model_type == "transformer":
            logits, target = model(batch)
            loss = label_smoothed_nll_loss(logits, target, epsilon=0.1)
            n_tokens = (target != PAD_ID).sum().item()
        else:
            logits, target = model(batch, teacher_forcing_ratio=tf_ratio)
            logits = logits.reshape(-1, logits.size(-1))
            target = target.reshape(-1)
            loss = ce_loss(logits, target)
            n_tokens = (target != PAD_ID).sum().item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

        if step % 100 == 0:
            print(
                f"Epoch {epoch_idx+1} Step {step}: "
                f"loss={loss.item():.4f}, tf_ratio={tf_ratio:.3f}"
            )

    return total_loss / max(total_tokens, 1)


def train(
    model_type: str,
    data_dir: str,
    lang_pair: str,
    spm_model: str,
    batch_size: int,
    epochs: int,
    save_dir: str,
    device: str,
):
    # For bilingual training lang_pair looks like "de-en".
    # For multilingual training we expect a folder named exactly `lang_pair`
    # (e.g. "multilingual") that contains {split}.src/.tgt.
    pair_dir = os.path.join(data_dir, lang_pair)
    train_src = os.path.join(pair_dir, "train.src")
    train_tgt = os.path.join(pair_dir, "train.tgt")
    valid_src = os.path.join(pair_dir, "valid.src")
    valid_tgt = os.path.join(pair_dir, "valid.tgt")

    sp = spm.SentencePieceProcessor(model_file=spm_model)
    vocab_size = sp.get_piece_size()

    train_loader = make_dataloader(
        train_src, train_tgt, spm_model, batch_size=batch_size, shuffle=True
    )
    valid_loader = make_dataloader(
        valid_src, valid_tgt, spm_model, batch_size=batch_size, shuffle=False
    )

    model = build_model(model_type, vocab_size=vocab_size).to(device)
    optimizer, scheduler = build_optim(model, model_type=model_type)

    os.makedirs(save_dir, exist_ok=True)
    best_val = math.inf
    best_path = os.path.join(save_dir, f"{lang_pair}.{model_type}.pt")

    for epoch in range(epochs):
        train_loss = run_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            epoch,
            epochs,
            model_type,
        )
        val_loss = run_epoch(
            model,
            valid_loader,
            optimizer,
            None,
            device,
            epoch,
            epochs,
            model_type,
        )
        print(
            f"[{model_type}] {lang_pair} Epoch {epoch+1}/{epochs}: "
            f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
        )
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": model.config.__dict__,
                    "model_type": model_type,
                    "spm_model": spm_model,
                },
                best_path,
            )
            print(f"  Saved best model to {best_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", choices=["seq2seq", "transformer"], required=True)
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/root/autodl-tmp/processed_wmt14",
    )
    parser.add_argument(
        "--lang-pair",
        type=str,
        required=True,
        help="language pair or dataset name (e.g. de-en or multilingual)",
    )
    parser.add_argument(
        "--spm-model",
        type=str,
        default="/root/autodl-tmp/processed_wmt14/spm/spm.model",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument(
        "--save-dir",
        type=str,
        default="checkpoints",
    )
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    train(
        model_type=args.model_type,
        data_dir=args.data_dir,
        lang_pair=args.lang_pair,
        spm_model=args.spm_model,
        batch_size=args.batch_size,
        epochs=args.epochs,
        save_dir=args.save_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
