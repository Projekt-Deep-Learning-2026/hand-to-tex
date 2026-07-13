import math

import torch
import torch.nn as nn


class TimestepEmbedder(nn.Module):
    LOG_CONST: float = 10000.0

    def __init__(self, hidden_dim: int, freq_embedding_size: int):
        super().__init__()

        self.freq_embed_size = freq_embedding_size
        self.hidden_dim = hidden_dim

        self.mlp = nn.Sequential(
            nn.Linear(in_features=self.freq_embed_size, out_features=self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:

        half_dim = self.freq_embed_size // 2

        emb = math.log(self.LOG_CONST) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb)

        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)

        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        return self.mlp(emb)
