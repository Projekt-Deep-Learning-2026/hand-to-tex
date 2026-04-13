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
            nn.Conv1d(in_channels, d_model, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=5, stride=2, padding=2),
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

    def _get_padding_mask(self, lengths: Tensor, max_len: int) -> Tensor:
        device = lengths.device
        mask = torch.arange(max_len, device=device).expand(
            lengths.size(0), max_len
        ) >= lengths.unsqueeze(1)
        return mask

    def _calc_downsampled_lengths(self, lengths: Tensor) -> Tensor:

        conv1_len = ((lengths + 2 * 2 - 5) // 2) + 1
        conv2_len = ((conv1_len + 2 * 2 - 5) // 2) + 1
        return conv2_len

    def forward(self, src: Tensor, src_lengths: Tensor, tgt: Tensor) -> Tensor:
        src_conv = src.transpose(1, 2)
        src_features = self.input_proj(src_conv).transpose(1, 2)

        src_emb = self.src_pe(src_features)

        new_src_lengths = self._calc_downsampled_lengths(src_lengths)
        src_key_padding_mask = self._get_padding_mask(new_src_lengths, src_features.size(1))

        tgt_emb = self.tgt_pe(self.tgt_tok_emb(tgt) * math.sqrt(self.d_model))
        tgt_key_padding_mask = tgt == self.pad_idx

        tgt_seq_len = tgt.size(1)
        tgt_causal_mask = nn.Transformer.generate_square_subsequent_mask(
            tgt_seq_len, device=tgt.device
        ).bool()

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

    def encode(
        self, src: torch.Tensor, src_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        src_conv = src.transpose(1, 2)

        src_features = self.input_proj(src_conv).transpose(1, 2)

        conv1_len = ((src_lengths + 2 * 2 - 5) // 2) + 1
        new_src_lengths = ((conv1_len + 2 * 2 - 5) // 2) + 1

        src_emb = self.src_pe(src_features)

        src_key_padding_mask = self._get_padding_mask(new_src_lengths, src_features.size(1))

        memory = self.transformer.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)

        return memory, src_key_padding_mask

    def decode(self, tgt: Tensor, memory: Tensor, memory_key_padding_mask: Tensor) -> Tensor:

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
