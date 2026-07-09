from torch import Tensor

type Features = Tensor
"""
Representation of a single sequence of points (strokes).
Dtype: torch.float32
Shape: [L, 4], where L is the sequence length

Feature indices:
    -- x coordinate (horizontal)
    -- y coordinate (vertical)
    -- dt / delta_t (normalized time difference from the previous point)
    -- is_new_stroke / pen_state (new stroke flag, 1.0 for new stroke, -1.0 in either case)
"""

type Batch = tuple[Tensor, Tensor]
"""
Represent batch of the generative datamodule.
Type: torch.Tensor(dtype=torch.float32), torch.Tensor(dtype=torch.bool)
Shape: ([B, L, 4], [B, L])

"""
