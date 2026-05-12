import math

import torch
from torch import Tensor, nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for batch-first transformer inputs.

    The module expects tensors shaped `(batch, seq_len, d_model)` and adds
    deterministic sinusoidal vectors to each position before dropout.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """Precompute sinusoidal encodings and register them as a buffer.

        Parameters
        ----------
        d_model:
            Embedding size of the model.
        dropout:
            Dropout probability applied after positional addition.
        max_len:
            Maximum sequence length covered by the precomputed table.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """Add position encodings to an input batch.

        Parameters
        ----------
        x:
            Input tensor of shape `(B, T, D)`.

        Returns
        -------
        Tensor
            Tensor with positional information added, same shape as input.
        """
        # seq_len = x.size(1)
        # x = x + self.pe[:, :seq_len, :]
        x = x + self.pe.narrow(1, 0, x.shape[1])
        return self.dropout(x)

    def forward_step(self, x: Tensor, step: int) -> Tensor:
        """Add position encoding for a single decoding step.

        Parameters
        ----------
        x:
            Input tensor of shape `(B, 1, D)`.
        step:
            Zero-based decoding step.

        Returns
        -------
        Tensor
            Positionalized tensor of shape `(B, 1, D)`.
        """
        step_tensor = torch.tensor([step], dtype=torch.long, device=x.device)
        pe_step = torch.index_select(self.pe.squeeze(0), dim=0, index=step_tensor).unsqueeze(0)

        x = x + pe_step
        return self.dropout(x)
