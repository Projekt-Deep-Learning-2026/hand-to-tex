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
        """Generate token sequences from input features.

        Parameters
        ----------
        src:
            Input feature tensor.
        src_lengths:
            Lengths of the input sequences.
        sos_idx:
            Start-of-sequence token id.
        eos_idx:
            End-of-sequence token id.
        max_len:
            Maximum generated sequence length.
        """

        raise NotImplementedError
