from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Final

import torch
from torch import Tensor
from torch.utils.data.dataset import Dataset

from hand_to_tex.datasets.ink_data import InkData
from hand_to_tex.utils import LatexVocab


class _HMEDatasetBase(Dataset, ABC):
    """Base class for Handwritten Math Expressions dataset"""

    EPS: Final[float] = 1e-6

    def __init__(
        self,
        root: Path | str,
        split: str,
        vocab: LatexVocab | None = None,
        transform: Callable[[Tensor], Tensor] | None = None,
    ):
        """Create HME dataset object

        Parameters
        ----------
        root : Path | str
            Directory of all dataset files
        split : str
            Name of the split, the path to .inkml files dir. will be root/split
        vocab : LatexVocab | None
            Vocabulary for tokenization, default `LatexVocab.default()`
        transform : (`Tensor` -> `Tensor`) | None
            Optional transformation applied to features before returning
        """
        self.data_path = Path(root, split)
        self.root = root
        self.split = split
        self.vocab = LatexVocab.default() if vocab is None else vocab
        self.transform = transform

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, _idx: int):
        pass

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
        if ink.traces == []:
            return torch.zeros((0, 10), dtype=torch.float32)

        traces = [torch.as_tensor(t, dtype=torch.float32) for t in ink.traces if len(t) > 0]

        trace_lengths = [trace.shape[0] for trace in traces]
        all_points = torch.cat(traces, dim=0)
        all_points = _HMEDatasetBase._normalise_data(all_points)
        norm_traces = all_points.split(trace_lengths)

        features_per_trace = []
        for trace in norm_traces:
            d_xyt = _HMEDatasetBase._trace_deltas(trace)
            dynamics = _HMEDatasetBase._trace_dynamics(d_xyt)

            is_stroke_start = torch.zeros(trace.shape[0], dtype=torch.float32, device=trace.device)
            is_stroke_start[0] = 1.0

            features_per_trace.append(
                torch.cat(
                    [
                        trace,
                        d_xyt,
                        dynamics,
                        is_stroke_start.unsqueeze(1),
                    ],
                    dim=1,
                )
            )

        return torch.cat(features_per_trace, dim=0)

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

        xy_range = (xy_max - xy_min).max() + _HMEDatasetBase.EPS
        xy_norm = (xyt[:, :2] - xy_min) / xy_range
        t_norm = xyt[:, 2:3] - t0

        return torch.cat([xy_norm, t_norm], dim=1)

    @staticmethod
    def _trace_deltas(xyt: Tensor) -> Tensor:
        """Compute `(dx, dy, dt)` for a single normalized trace.

        The first point in every trace has zero deltas by definition.
        """
        deltas = torch.zeros_like(xyt)
        deltas[1:] = xyt[1:] - xyt[:-1]
        return deltas

    @staticmethod
    def _trace_dynamics(d_xyt: Tensor) -> Tensor:
        """Extract dynamic handwriting features from one trace deltas.

        Parameters
        ----------
        d_xyt : Tensor
            Tensor `(N, 3)` with `(dx, dy, dt)`.

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
            dt > _HMEDatasetBase.EPS,
            dist / dt,
            torch.zeros_like(dist),
        )

        ux = torch.where(
            dist > _HMEDatasetBase.EPS,
            dx / dist,
            torch.zeros_like(dx),
        )
        uy = torch.where(
            dist > _HMEDatasetBase.EPS,
            dy / dist,
            torch.zeros_like(dy),
        )

        cross = ux[:-1] * uy[1:] - uy[:-1] * ux[1:]
        dot = ux[:-1] * ux[1:] + uy[:-1] * uy[1:]
        dtheta = torch.atan2(cross, dot)

        curve = torch.zeros_like(dist)
        curve[1:] = torch.where(
            dist[1:] > _HMEDatasetBase.EPS,
            dtheta / dist[1:],
            torch.zeros_like(dtheta),
        )

        dspeed = torch.zeros_like(speed)
        dspeed[1:] = speed[1:] - speed[:-1]
        acc_tan = torch.where(
            dt > _HMEDatasetBase.EPS,
            dspeed / dt,
            torch.zeros_like(dspeed),
        )

        return torch.cat(
            [
                speed.unsqueeze(1),
                curve.unsqueeze(1),
                acc_tan.unsqueeze(1),
            ],
            dim=1,
        )


class HMEDatasetRaw(_HMEDatasetBase):
    """HME Dataset that operates on raw .inkml files"""

    def __init__(
        self,
        root: Path | str,
        split: str,
        vocab: LatexVocab | None = None,
        transform: Callable[[Tensor], Tensor] | None = None,
    ):
        super().__init__(root=root, split=split, vocab=vocab, transform=transform)
        self.data_path = Path(root, split)
        self.filenames = sorted(self.data_path.rglob("*.inkml"))

    def __len__(self) -> int:
        return len(self.filenames)

    def __repr__(self) -> str:
        return f"HMEDataset(root={self.root!r}, split={self.split}, n_samples={len(self)})"

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        ink = InkData.load(self.filenames[idx])
        features = _HMEDatasetBase.extract_features(ink)
        truth = ink.tex_norm
        tokens = self.vocab.encode_expr(truth)

        if self.transform:
            features = self.transform(features)

        return features, torch.tensor(tokens, dtype=torch.long)


class HMEDatasetPreprocessed(_HMEDatasetBase):
    """HME Dataset that operates on preprocessed data stored in .pt"""

    def __init__(
        self,
        root: Path | str,
        split: str,
        vocab: LatexVocab | None = None,
        transform: Callable[[Tensor], Tensor] | None = None,
    ):
        super().__init__(root=root, split=split, vocab=vocab, transform=transform)
        data_path = Path(root, split + ".pt")
        self.data = torch.load(data_path)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self.data[idx]
