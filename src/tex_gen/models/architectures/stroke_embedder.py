import math

import torch
import torch.nn as nn

from tex_gen.types import Features


class StrokeEmbedder(nn.Module):
    def __init__(self, in_channels: int, hidden_size: int, max_seq_len: int):
        super().__init__()

        self.input_proj = nn.Linear(in_features=in_channels, out_features=hidden_size)

        pe = self._generate_pe(max_len=max_seq_len, d_model=hidden_size)
        self.register_buffer("pe", pe, persistent=False)

    @staticmethod
    def _generate_pe(max_len: int, d_model: int) -> torch.Tensor:

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe.unsqueeze(0)

    def forward(self, x: Features) -> torch.Tensor:

        seq_len = x.size(0)

        x_proj = self.input_proj(x)

        x_embed = x_proj + self.pe[:, :seq_len, :]  # type: ignore

        return x_embed
