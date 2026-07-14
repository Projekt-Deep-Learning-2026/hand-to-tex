import torch
import torch.nn as nn

from tex_gen.models.architectures.dit_block import DiTBlock
from tex_gen.models.architectures.stroke_embedder import StrokeEmbedder
from tex_gen.models.architectures.timestep_embedder import TimestepEmbedder
from tex_gen.types import Features


class DiT(nn.Module):
    def __init__(
        self, in_channels: int, hidden_dim: int, depth: int, num_heads: int, max_seq_len: int
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len

        self.stroke_embedder = StrokeEmbedder(
            in_channels=self.in_channels, hidden_dim=self.hidden_dim, max_seq_len=self.max_seq_len
        )
        self.ts_embedder = TimestepEmbedder(hidden_dim=self.hidden_dim, freq_embedding_size=256)

        self.blocks = nn.ModuleList(
            [
                DiTBlock(hidden_dim=self.hidden_dim, num_heads=self.num_heads)
                for _ in range(self.depth)
            ]
        )

        self.final_norm = nn.LayerNorm(self.hidden_dim, elementwise_affine=False)
        self.final_modulator = nn.Sequential(
            nn.SiLU(), nn.Linear(self.hidden_dim, 2 * self.hidden_dim)
        )
        nn.init.zeros_(self.final_modulator[1].weight)  # type: ignore
        nn.init.zeros_(self.final_modulator[1].bias)  # type: ignore

        self.output_proj = nn.Linear(self.hidden_dim, self.in_channels)

    def forward(self, x: Features, timesteps: torch.Tensor, padding_mask: torch.Tensor) -> Features:

        x_emb = self.stroke_embedder(x)
        t_emb = self.ts_embedder(timesteps)

        for block in self.blocks:
            x_emb = block(x_emb, t_emb, padding_mask)

        shift, scale = self.final_modulator(t_emb).chunk(2, dim=1)

        shift, scale = shift.unsqueeze(1), scale.unsqueeze(1)

        x_emb = self.final_norm(x_emb) * (1 + scale) + shift

        pred_noise = self.output_proj(x_emb)

        return pred_noise
