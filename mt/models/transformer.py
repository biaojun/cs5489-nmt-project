from dataclasses import dataclass

import torch
import torch.nn as nn

from mt.data import PAD_ID


@dataclass
class TransformerConfig:
    vocab_size: int
    d_model: int = 512
    nhead: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    dim_feedforward: int = 2048
    dropout: float = 0.1


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.src_embed = nn.Embedding(config.vocab_size, config.d_model, padding_idx=PAD_ID)
        self.tgt_embed = nn.Embedding(config.vocab_size, config.d_model, padding_idx=PAD_ID)
        self.pos_enc = PositionalEncoding(config.d_model, config.dropout)
        self.transformer = nn.Transformer(
            d_model=config.d_model,
            nhead=config.nhead,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
        )
        self.generator = nn.Linear(config.d_model, config.vocab_size)

    def _generate_square_subsequent_mask(self, sz: int, device):
        mask = torch.triu(torch.ones(sz, sz, device=device) == 1, diagonal=1)
        mask = mask.float().masked_fill(mask, float("-inf"))
        return mask

    def forward(self, batch):
        src = batch.src
        tgt_input = batch.tgt_input
        src_key_padding_mask = src.eq(PAD_ID)
        tgt_key_padding_mask = tgt_input.eq(PAD_ID)
        tgt_mask = self._generate_square_subsequent_mask(tgt_input.size(1), src.device)

        src_emb = self.pos_enc(self.src_embed(src))
        tgt_emb = self.pos_enc(self.tgt_embed(tgt_input))

        memory = self.transformer.encoder(
            src_emb, src_key_padding_mask=src_key_padding_mask
        )
        output = self.transformer.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        logits = self.generator(output)
        return logits, batch.tgt_output

    def greedy_decode(self, src, max_len: int = 100):
        src_key_padding_mask = src.eq(PAD_ID)
        src_emb = self.pos_enc(self.src_embed(src))
        memory = self.transformer.encoder(
            src_emb, src_key_padding_mask=src_key_padding_mask
        )
        batch_size = src.size(0)
        ys = torch.full((batch_size, 1), 1, dtype=torch.long, device=src.device)
        for _ in range(max_len):
            tgt_key_padding_mask = ys.eq(PAD_ID)
            tgt_mask = self._generate_square_subsequent_mask(ys.size(1), src.device)
            tgt_emb = self.pos_enc(self.tgt_embed(ys))
            out = self.transformer.decoder(
                tgt_emb,
                memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=src_key_padding_mask,
            )
            prob = self.generator(out[:, -1])
            next_word = prob.argmax(dim=-1)
            ys = torch.cat([ys, next_word.unsqueeze(1)], dim=1)
        return ys
