import math

import torch
import torch.nn as nn
from torch import Tensor


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        seq_len = x.size(1)
        x = x + self.pe[:seq_len, :].unsqueeze(0)  # type: ignore
        return self.dropout(x)


class ExperimentalTransformer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        vocab_size: int,
        pad_idx: int,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.d_model = d_model

        self.input_proj = nn.Sequential(
            nn.Conv1d(in_channels, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )

        self.tgt_tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)

        self.src_pe = PositionalEncoding(d_model, dropout)
        self.tgt_pe = PositionalEncoding(d_model, dropout)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.fc_out = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _get_padding_mask(self, lengths: Tensor, max_len: int) -> Tensor:
        device = lengths.device
        mask = torch.arange(max_len, device=device).expand(
            lengths.size(0), max_len
        ) >= lengths.unsqueeze(1)
        return mask

    def forward(self, src: Tensor, src_lengths: Tensor, tgt: Tensor) -> Tensor:
        src_conv = src.transpose(1, 2)
        src_features = self.input_proj(src_conv)
        src_features = src_features.transpose(1, 2)  # Wracamy do [B, S, C]

        src_emb = self.src_pe(src_features)

        src_key_padding_mask = self._get_padding_mask(src_lengths, src.size(1))

        tgt_emb = self.tgt_pe(self.tgt_tok_emb(tgt) * math.sqrt(self.d_model))

        tgt_key_padding_mask = tgt == self.pad_idx

        tgt_seq_len = tgt.size(1)
        tgt_causal_mask = nn.Transformer.generate_square_subsequent_mask(
            tgt_seq_len, device=tgt.device
        )

        output = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            src_mask=None,
            tgt_mask=tgt_causal_mask,
            memory_mask=None,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )

        return self.fc_out(output)

    def encode(self, src: Tensor, src_lengths: Tensor) -> tuple[Tensor, Tensor]:
        """Oblicza reprezentację trajektorii tylko RAZ."""
        src_conv = src.transpose(1, 2)
        src_features = self.input_proj(src_conv).transpose(1, 2)

        src_emb = self.src_pe(src_features)
        src_key_padding_mask = self._get_padding_mask(src_lengths, src.size(1))

        memory = self.transformer.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
        return memory, src_key_padding_mask

    def decode(self, tgt: Tensor, memory: Tensor, memory_key_padding_mask: Tensor) -> Tensor:
        """Szybko generuje następny token wykorzystując Cache z pamięci obrazka."""
        tgt_emb = self.tgt_pe(self.tgt_tok_emb(tgt) * math.sqrt(self.d_model))

        tgt_causal_mask = nn.Transformer.generate_square_subsequent_mask(
            tgt.size(1), device=tgt.device
        )
        tgt_key_padding_mask = tgt == self.pad_idx

        out = self.transformer.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.fc_out(out)
