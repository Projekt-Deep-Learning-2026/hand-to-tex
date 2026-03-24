from pathlib import Path
import torch
from torch.utils.data.dataset import Dataset
from torch import Tensor
from typing import Optional, Callable

from .ink_data import InkData, Trace
from ..utils import LatexVocab


class HMEDataset(Dataset):
    """Handwritten Mathematical Expressions dataset for processing
    `.inkml` files
    """
    EPS: float = 1e-8

    def __init__(
                self,
                root: Path | str,
                vocab: Optional[LatexVocab] = None,
                transform: Optional[Callable] = None,
            ):
        """Creates the dataset

        Parameters
        ----------
        root : Path | str
            Directory of all dataset files
        vocab : LatexVocab | None
            Vocabulary for tokenization, default `LatexVocab.default()`
        transform : Callable | None
            #TODO
        """

        self.root = Path(root)
        self.filenames = sorted(list(self.root.rglob('*.inkml')))

        self.vocab = LatexVocab.default() if vocab is None else vocab
        self.transform = transform

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int):
        # p = self.filenames[idx]
        return InkData.load(self.filenames[idx])
        # raise NotImplementedError

    @staticmethod
    def extract_features(ink: InkData) -> torch.Tensor:
        """Extracts features of an InkData object

        Returns
        -------
        torch.Tensor of shape `(n, 11)` with features:
            - x_norm, y_norm
                Normalised coordinates of a trace point
            - t_rel
                Relative timestamp (shifted `t := t - t0`)
            - dx, dy, dt
                Change in position and time between every TracePoint
            - speed
                Pen speed `sqrt(dx^2 + dy^2) / (dt + EPS)`
            - dir_x, dir_y
                Normalized writing direction vector
            - is_stroke_start, is_stroke_end
                Binary indicators of stroke boundaries
        """
        xyt, t_idxs = HMEDataset._flatten_traces(ink.traces)

        xyt = HMEDataset._normalise_data(xyt)
        d_xyt, same_trace_prev = HMEDataset._to_deltas(xyt, t_idxs)

        dist = torch.hypot(d_xyt[:, 0], d_xyt[:, 1])
        speed = torch.where(
            d_xyt[:, 2] > 0,
            dist / (d_xyt[:, 2] + HMEDataset.EPS),
            torch.zeros_like(dist)
        )
        dir_x = torch.where(
            dist > 0,
            d_xyt[:, 0] / (dist + HMEDataset.EPS),
            torch.zeros_like(d_xyt[:, 0])
        )
        dir_y = torch.where(
            dist > 0,
            d_xyt[:, 1] / (dist + HMEDataset.EPS),
            torch.zeros_like(d_xyt[:, 1])
        )

        is_stroke_start = (~same_trace_prev).float()
        is_stroke_end = (~torch.roll(same_trace_prev, -1)).float()

        return torch.stack([
            xyt,
            d_xyt,
            speed.unsqueeze(1),
            dir_x.unsqueeze(1),
            dir_y.unsqueeze(1),
            is_stroke_start.unsqueeze(1),
            is_stroke_end.unsqueeze(1),
        ], dim=1)

    @staticmethod
    def _flatten_traces(traces: list[Trace]) -> tuple[Tensor, Tensor]:
        """Returns pair of tensors : (x, y, t) points and trace indexes
        for InkData traces

        Parameters
        ----------
        traces : Traces
            List of Trace elements obtained from `InkData.traces`
        """
        xyt = torch.cat(
            [torch.as_tensor(t, dtype=torch.float32) for t in traces],
            dim=0
        )
        trace_idxs = torch.repeat_interleave(
            torch.arange(len(traces)),
            torch.as_tensor([len(t) for t in traces], dtype=torch.int64)
        )
        return xyt, trace_idxs

    @staticmethod
    def _normalise_data(xyt: Tensor) -> Tensor:
        """Perform normalization on `(x, y, t)` points tensor.
        Normalization moves (x, y) -> [0, 1]x[0, 1], and t := t - t0 where
        t0 is first timestamp recorded in input tensor

        Parameters
        ----------
        xyt : Tensor
            Tensor containing points `(x, y, t)`
        """
        xy = xyt[:, :2]
        xy_min = xy.min(dim=0, keepdim=True).values
        xy_max = xy.max(dim=0, keepdim=True).values
        t0 = xyt[:, 2].min(dim=0, keepdim=True).values

        xyt[:, :2] = (xy - xy_min) / (xy_max - xy_min + HMEDataset.EPS)
        xyt[:, 2] -= t0

        return xyt

    @staticmethod
    def _to_deltas(xyt: Tensor, t_idxs: Tensor) -> tuple[Tensor, Tensor]:
        """Converts tensor of points into tensor of changes,
        `(x, y, t)` -> `(dx, dy, dt)`
        and binary array that indicates whether point on this index belongs
        to the same trace as previous one or not
        """
        deltas = torch.zeros_like(xyt)
        deltas[1:] = xyt[1:] - xyt[:-1]

        same_prev = torch.zeros_like(xyt[:, 0], dtype=torch.bool)
        same_prev[1:] = t_idxs[1:] == t_idxs[:-1]

        deltas *= same_prev.unsqueeze(1)

        return deltas, same_prev


if __name__ == "__main__":

    d = HMEDataset('data/train')
    ink = d[4]
    HMEDataset.extract_features(ink)
