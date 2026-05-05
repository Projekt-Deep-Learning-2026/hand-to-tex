from collections.abc import Callable
from typing import TYPE_CHECKING

from torch import Tensor

if TYPE_CHECKING:
    from hand_to_tex.models.components.base import BaseDecoderModel

# Ink Data types
type TracePoint = tuple[float, float, float]
"""A single recorded point from a pen trajectory in the format `(x, y, timestamp)`."""

type Trace = list[TracePoint]
"""A continuous pen stroke consistently drawn from pen-down to pen-up."""

type Traces = list[Trace]
"""A collection of strokes making up a complete handwritten expression."""

# Tensor types
type TensorF32 = Tensor
"""PyTorch Tensor with `dtype=torch.float32` (single-precision float)."""

type TensorF16 = Tensor
"""PyTorch Tensor with `dtype=torch.float16` (half-precision float)."""

type TensorI16 = Tensor
"""PyTorch Tensor with `dtype=torch.int16` (16-bit integer)."""

type TensorI32 = Tensor
"""PyTorch Tensor with `dtype=torch.int32` (32-bit integer)."""

type TensorLong = Tensor
"""PyTorch Tensor with `dtype=torch.int64` (64-bit integer)"""

type TensorBool = Tensor
"""PyTorch Tensor with `dtype=torch.bool` (boolean mask tensor)."""

type Transformation[TensorType: Tensor] = Callable[[TensorType], TensorType]
"""A callable function or object that applies a transformation module to a given tensor."""

# Model Interface Types
type DecoderModel = "BaseDecoderModel"
"""Model implementing the autoregressive `generate()` interface."""

# KV-cache Types
type LayerKVCache = dict[str, Tensor]
"""Per-layer KV-cache with projected tensors (e.g., `self_k`, `self_v`, `mem_k`, `mem_v`)."""

type DecoderKVCache = dict[str, int | list[LayerKVCache]]
"""Decoder KV-cache storing the current step and per-layer cache tensors."""

# Feature Types
type Features = TensorF32
"""2D feature tensor extracted from traces. `N` points with `F` features. Expected shape: `(N, F)`."""

type BatchedFeatures = TensorF32
"""Feature tensor with a batch dimension and sequence padding. Expected shape: `(B, max_N, F)`."""

type FeatureLengths = TensorLong
"""1D tensor of true (unpadded) sequence lengths for the features in a batch. Expected shape: `(B,)`."""

# Token Types
type Tokens = TensorLong
"""1D tensor of integer token IDs representing a math expression. Expected shape: `(L,)`."""

type BatchedTokens = TensorLong
"""Token tensor with a batch dimension and sequence padding. Expected shape: `(B, max_L)`."""

type TokenLengths = TensorLong
"""1D tensor of true (unpadded) sequence lengths for the tokens in a batch. Expected shape: `(B,)`."""

# Batch Types
type Sample = tuple[Features, Tokens]
"""A single dataset sample, pair `[Features, Tokens]`"""

type Batch = tuple[BatchedFeatures, FeatureLengths, BatchedTokens, TokenLengths]
"""Single batch type contains `(BatchedFeatures, FeatureLenghts, BatchedTokens, TokenLengths)`"""
