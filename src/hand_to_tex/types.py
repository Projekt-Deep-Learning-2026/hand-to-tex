from collections.abc import Callable

from torch import Tensor

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

type Features = TensorF32
"""2D feature tensor of shape `(N, F)` extracted from traces. `N` points with `F` features."""

type Tokens = TensorLong
"""1D tensor of integer token IDs representing a math expression."""

type Sample = tuple[Features, Tokens]
"""A single dataset sample, pair `[Features, Tokens]`"""
