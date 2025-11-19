from dataclasses import dataclass

import torch
import torch.nn as nn

from mt.data import PAD_ID


@dataclass
class Seq2SeqConfig:
    vocab_size: int
    emb_dim: int = 512
    hidden_size: int = 512
    num_layers: int = 1
    dropout: float = 0.3


class Encoder(nn.Module):
    def __init__(self, config: Seq2SeqConfig):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.emb_dim, padding_idx=PAD_ID)
        self.lstm = nn.LSTM(
            input_size=config.emb_dim,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            bidirectional=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            batch_first=True,
        )

    def forward(self, src, src_lens):
        embedded = self.embedding(src)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, src_lens.cpu(), batch_first=True, enforce_sorted=False
        )
        outputs, (h, c) = self.lstm(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        return outputs, (h, c)


class LuongAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.W = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, query, values, mask):
        # query: [B, H], values: [B, T, H]
        score = torch.bmm(values, self.W(query).unsqueeze(-1)).squeeze(-1)
        score = score.masked_fill(mask == 0, -1e9)
        attn_weights = torch.softmax(score, dim=-1)
        context = torch.bmm(attn_weights.unsqueeze(1), values).squeeze(1)
        return context, attn_weights


class Decoder(nn.Module):
    def __init__(self, config: Seq2SeqConfig):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.emb_dim, padding_idx=PAD_ID)
        self.lstm = nn.LSTM(
            input_size=config.emb_dim,
            hidden_size=config.hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.attn = LuongAttention(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.fc_out = nn.Linear(config.hidden_size * 2, config.vocab_size)

    def forward_step(self, input_tokens, hidden, enc_outputs, src_mask):
        emb = self.dropout(self.embedding(input_tokens.unsqueeze(1)))
        output, hidden = self.lstm(emb, hidden)
        output = output.squeeze(1)
        context, _ = self.attn(output, enc_outputs, src_mask)
        concat = torch.cat([output, context], dim=-1)
        logits = self.fc_out(concat)
        return logits, hidden


class Seq2SeqModel(nn.Module):
    def __init__(self, config: Seq2SeqConfig):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.hidden_proj = nn.Linear(config.hidden_size * 2, config.hidden_size)

    def forward(self, batch, teacher_forcing_ratio: float = 1.0):
        src, src_lens = batch.src, batch.src_lens
        tgt_input, tgt_output = batch.tgt_input, batch.tgt_output
        batch_size, tgt_len = tgt_input.size()

        enc_outputs, (h, c) = self.encoder(src, src_lens)
        # combine bidirectional states
        h_cat = torch.cat([h[-2], h[-1]], dim=-1)
        c_cat = torch.cat([c[-2], c[-1]], dim=-1)
        h0 = self.hidden_proj(h_cat).unsqueeze(0)
        c0 = self.hidden_proj(c_cat).unsqueeze(0)
        hidden = (h0, c0)

        src_mask = (src != PAD_ID).float()

        logits = []
        input_tokens = tgt_input[:, 0]
        for t in range(1, tgt_len + 1):
            step_logits, hidden = self.decoder(
                input_tokens, hidden, enc_outputs, src_mask
            )
            logits.append(step_logits.unsqueeze(1))
            if t == tgt_len:
                break
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            if teacher_force:
                input_tokens = tgt_input[:, t]
            else:
                input_tokens = step_logits.argmax(dim=-1)

        logits = torch.cat(logits, dim=1)
        return logits, tgt_output

    def greedy_decode(self, src, src_lens, max_len: int = 100):
        enc_outputs, (h, c) = self.encoder(src, src_lens)
        h_cat = torch.cat([h[-2], h[-1]], dim=-1)
        c_cat = torch.cat([c[-2], c[-1]], dim=-1)
        h0 = self.hidden_proj(h_cat).unsqueeze(0)
        c0 = self.hidden_proj(c_cat).unsqueeze(0)
        hidden = (h0, c0)
        src_mask = (src != PAD_ID).float()

        batch_size = src.size(0)
        outputs = torch.full((batch_size, 1), fill_value=1, dtype=torch.long, device=src.device)
        input_tokens = torch.full((batch_size,), fill_value=1, dtype=torch.long, device=src.device)

        for _ in range(max_len):
            logits, hidden = self.decoder(input_tokens, hidden, enc_outputs, src_mask)
            next_tokens = logits.argmax(dim=-1)
            outputs = torch.cat([outputs, next_tokens.unsqueeze(1)], dim=1)
            input_tokens = next_tokens
        return outputs

