import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from tex_gen.types import Features


class PaddedMSELoss(nn.Module):
    """
    Calculates MSE loss including the padding mask
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, inp: Features, tgt: Features, mask: Tensor) -> Tensor:

        loss_unreduced = F.mse_loss(inp, tgt, reduction="none")
        mask_expanded = mask.unsqueeze(-1).float()

        loss_masked = loss_unreduced * mask_expanded

        valid_elems = mask.sum() * inp.shape[-1]

        return loss_masked / (valid_elems + self.eps)
