import torch
import torch.nn as nn


class DiTBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)

        self.attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim, num_heads=self.num_heads, batch_first=True
        )

        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 4),
            nn.GELU(),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(self.hidden_dim, self.hidden_dim * 6)
        )
        nn.init.zeros_(self.adaLN_modulation[1].weight)  # type: ignore
        nn.init.zeros_(self.adaLN_modulation[1].bias)  # type: ignore

    def forward(self, x: torch.Tensor, c: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
            c
        ).chunk(6, dim=1)

        shift_msa, scale_msa, gate_msa = (
            shift_msa.unsqueeze(1),
            scale_msa.unsqueeze(1),
            gate_msa.unsqueeze(1),
        )
        shift_mlp, scale_mlp, gate_mlp = (
            shift_mlp.unsqueeze(1),
            scale_mlp.unsqueeze(1),
            gate_mlp.unsqueeze(1),
        )

        key_padding_mask = ~padding_mask

        x_modulated = self.norm1(x) * (1 + scale_msa) + shift_msa

        attn_out, _ = self.attn(
            query=x_modulated,
            key=x_modulated,
            value=x_modulated,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + gate_msa * attn_out

        x_modulated = self.norm2(x) * (1 + scale_mlp) + shift_mlp
        ffn_out = self.ffn(x_modulated)

        x = x + gate_mlp * ffn_out

        return x
