from __future__ import annotations

from abc import ABC, abstractmethod

import torch.nn as nn
from torch import Tensor


class BaseDecoderModel(nn.Module, ABC):
    """Base interface for handwriting-to-LaTeX models with autoregressive generate()."""

    @abstractmethod
    def generate(
        self,
        src: Tensor,
        src_lengths: Tensor,
        *,
        sos_idx: int,
        eos_idx: int,
        max_len: int,
    ) -> Tensor:
        """Generate token ids from input features."""
        raise NotImplementedError
