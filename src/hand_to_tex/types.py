from collections.abc import Callable

from torch import Tensor

# Ink Data types
type TracePoint = tuple[float, float, float]
type Trace = list[TracePoint]
type Traces = list[Trace]

# Tensor types
type TensorF32 = Tensor
"""Tensor with `dtype=torch.float32`"""
type TensorF16 = Tensor
type TensorI16 = Tensor
type TensorI32 = Tensor
type TensorLong = Tensor
type TensorBool = Tensor

type Transformation[TensorType: Tensor] = Callable[[TensorType], TensorType]
"""Transforms given tensor into another one"""

type Features = TensorF32
"""Feature tensor"""

type Tokens = TensorLong
"""Token tensor"""

type Sample = tuple[Features, Tokens]
"""Single sample [fts, tokens]"""
