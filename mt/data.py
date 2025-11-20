import os
from dataclasses import dataclass
from typing import List, Tuple

import sentencepiece as spm
import torch
from torch.utils.data import Dataset, DataLoader


PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3


class ParallelDataset(Dataset):
    def __init__(self, src_path: str, tgt_path: str, sp_model: str):
        self.src_lines = self._read_file(src_path)
        self.tgt_lines = self._read_file(tgt_path)
        assert len(self.src_lines) == len(
            self.tgt_lines
        ), "Source and target must have same number of lines"
        self.sp = spm.SentencePieceProcessor(model_file=sp_model)

    @staticmethod
    def _read_file(path: str) -> List[str]:
        with open(path, "r", encoding="utf-8") as f:
            return [line.rstrip("\n") for line in f if line.strip()]

    def __len__(self) -> int:
        return len(self.src_lines)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        src_tokens = self.src_lines[idx].split()
        tgt_tokens = self.tgt_lines[idx].split()
        # .src/.tgt 已经是 SentencePiece 子词序列（字符串），这里只需要映射到 id。
        src_ids = [BOS_ID] + [self.sp.piece_to_id(t) for t in src_tokens] + [EOS_ID]
        tgt_ids = [BOS_ID] + [self.sp.piece_to_id(t) for t in tgt_tokens] + [EOS_ID]
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(
            tgt_ids, dtype=torch.long
        )


@dataclass
class Batch:
    src: torch.Tensor
    src_lens: torch.Tensor
    tgt_input: torch.Tensor
    tgt_output: torch.Tensor
    tgt_lens: torch.Tensor


def collate_batch(samples: List[Tuple[torch.Tensor, torch.Tensor]]) -> Batch:
    src_seqs, tgt_seqs = zip(*samples)
    src_lens = torch.tensor([len(s) for s in src_seqs], dtype=torch.long)
    tgt_lens = torch.tensor([len(t) for t in tgt_seqs], dtype=torch.long)

    max_src = src_lens.max().item()
    max_tgt = tgt_lens.max().item()

    pad = PAD_ID
    batch_size = len(samples)
    src_batch = torch.full((batch_size, max_src), pad, dtype=torch.long)
    tgt_batch = torch.full((batch_size, max_tgt), pad, dtype=torch.long)

    for i, (s, t) in enumerate(zip(src_seqs, tgt_seqs)):
        src_batch[i, : len(s)] = s
        tgt_batch[i, : len(t)] = t

    tgt_input = tgt_batch[:, :-1]
    tgt_output = tgt_batch[:, 1:]
    tgt_lens = tgt_lens - 1

    return Batch(
        src=src_batch,
        src_lens=src_lens,
        tgt_input=tgt_input,
        tgt_output=tgt_output,
        tgt_lens=tgt_lens,
    )


def make_dataloader(
    src_path: str,
    tgt_path: str,
    sp_model: str,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 2,
) -> DataLoader:
    dataset = ParallelDataset(src_path, tgt_path, sp_model)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_batch,
    )
