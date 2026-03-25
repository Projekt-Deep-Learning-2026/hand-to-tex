from pathlib import Path
import torch
from torch.utils.data.dataset import Dataset
from torch import Tensor
from typing import Callable, Final

from .ink_data import InkData
from ..utils import LatexVocab


class HMEDataset(Dataset):
    """Handwritten Mathematical Expressions dataset for processing
    `.inkml` files
    """
    EPS: Final[float] = 1e-6

    def __init__(
                self,
                root: Path | str,
                vocab: LatexVocab | None = None,
                transform: Callable[[Tensor], Tensor] | None = None,
            ):
        """Creates the dataset

        Parameters
        ----------
        root : Path | str
            Directory of all dataset files
        vocab : LatexVocab | None
            Vocabulary for tokenization, default `LatexVocab.default()`
        transform : (Callable: `Tensor` -> `Tensor`) | None
            Optional transformation applied to features before returning.
        """

        self.root = Path(root)
        self.filenames = sorted(self.root.rglob('*.inkml'))

        self.vocab = LatexVocab.default() if vocab is None else vocab
        self.transform = transform

    def __len__(self) -> int:
        return len(self.filenames)

    def __repr__(self) -> str:
        return f"HMEDataset(root={self.root!r}, n_samples={len(self)})"

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        ink = InkData.load(self.filenames[idx])
        features = HMEDataset.extract_features(ink)
        truth = ink.tex_norm
        tokens = self.vocab.encode_expr(truth)

        if self.transform:
            features = self.transform(features)

        return features, torch.tensor(tokens, dtype=torch.long)

    @staticmethod
    def extract_features(ink: InkData) -> Tensor:
        """Extracts features of an InkData object

        Returns
        -------
        torch.Tensor of shape `(N, 10)` with features:
            - x_norm, y_norm
                Normalised coordinates of a trace point
            - t_rel
                Relative timestamp (shifted `t := t - t0`)
            - dx, dy, dt
                Change in position and time between every TracePoint
            - speed
                Pen speed `sqrt(dx^2 + dy^2) / (dt + EPS)`
            - curvature
                Signed local curvature `dtheta / dist`
            - acc_tan
                Tangential acceleration `dspeed / dt`
            - is_stroke_start
                Binary indicators of stroke boundaries
        """
        xyt, t_idxs = HMEDataset._flatten_traces(ink.traces)

        xyt = HMEDataset._normalise_data(xyt)
        d_xyt, same_trace_prev = HMEDataset._to_deltas(xyt, t_idxs)

        dynamics = HMEDataset._dynamics(d_xyt, same_trace_prev)

        is_stroke_start = (~same_trace_prev).float()

        return torch.cat([
            xyt,
            d_xyt,
            dynamics,
            is_stroke_start.unsqueeze(1),
        ], dim=1)

    @staticmethod
    def _flatten_traces(traces: list) -> tuple[Tensor, Tensor]:
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
        Normalization uses uniform scaling to preserve aspect ratio,
        and shifts time so that `t := t - t0`.

        Parameters
        ----------
        xyt : Tensor
            Tensor containing points `(x, y, t)`

        Returns
        -------
        Tensor
            Shape `(N, 3)` with normalized (x, y, t)
        """
        xy_min = xyt[:, :2].min(dim=0, keepdim=True).values
        xy_max = xyt[:, :2].max(dim=0, keepdim=True).values
        t0 = xyt[:, 2:3].min(dim=0, keepdim=True).values

        xy_range = (xy_max - xy_min).max() + HMEDataset.EPS
        xy_norm = (xyt[:, :2] - xy_min) / xy_range
        t_norm = xyt[:, 2:3] - t0

        return torch.cat([xy_norm, t_norm], dim=1)

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

    @staticmethod
    def _dynamics(d_xyt: Tensor, same_prev: Tensor) -> Tensor:
        """Extract dynamic handwriting features from deltas.

        Parameters
        ----------
        d_xyt : Tensor
            Tensor `(N, 3)` with `(dx, dy, dt)`.
        same_prev : Tensor
            Boolean tensor `(N,)`, where `same_prev[i] == True` iff point `i`
            belongs to the same stroke as point `i-1`.

        Returns
        -------
        Tensor
            Tensor `(N, 3)` with columns:
            `(speed, curve, acc_tan)`.
        """
        dx = d_xyt[:, 0]
        dy = d_xyt[:, 1]
        dt = d_xyt[:, 2]

        dist = torch.hypot(dx, dy)

        speed = torch.where(
            dt > HMEDataset.EPS,
            dist / dt,
            torch.zeros_like(dist),
        )

        ux = torch.where(
            dist > HMEDataset.EPS,
            dx / dist,
            torch.zeros_like(dx),
        )
        uy = torch.where(
            dist > HMEDataset.EPS,
            dy / dist,
            torch.zeros_like(dy),
        )

        cross = ux[:-1] * uy[1:] - uy[:-1] * ux[1:]
        dot = ux[:-1] * ux[1:] + uy[:-1] * uy[1:]
        dtheta = torch.atan2(cross, dot)

        curve = torch.zeros_like(dist)
        curve = same_prev[1:] & same_prev[:-1] & (dist[1:] > HMEDataset.EPS)
        curve[1:] = torch.where(
            curve,
            dtheta / dist[1:],
            torch.zeros_like(dtheta),
        )

        dspeed = torch.zeros_like(speed)
        dspeed[1:] = speed[1:] - speed[:-1]
        acc_tan = torch.where(
            same_prev & (dt > HMEDataset.EPS),
            dspeed / dt,
            torch.zeros_like(dspeed),
        )

        return torch.cat([
            speed.unsqueeze(1),
            curve.unsqueeze(1),
            acc_tan.unsqueeze(1),
        ], dim=1)
