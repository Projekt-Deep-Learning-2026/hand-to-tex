from abc import ABC, abstractmethod
from pathlib import Path
from typing import Final

import torch
from torch.utils.data.dataset import Dataset

from hand_to_tex.datasets.ink_data import InkData
from hand_to_tex.types import Features, Sample, TensorF32, Transformation
from hand_to_tex.utils import LatexVocab, logger


class _HMEDatasetBase(Dataset, ABC):
    """Base class for Handwritten Math Expressions dataset"""

    EPS: Final[float] = 1e-6
    FEATURES: Final[int] = 12

    def __init__(
        self,
        root: Path | str,
        split: str,
        vocab: LatexVocab | None = None,
        transform: Transformation[Features] | None = None,
    ):
        """Create HME dataset object

        Parameters
        ----------
        root
            Directory of all dataset files
        split
            Name of the split, the path to .inkml files dir. will be root/split
        vocab
            Vocabulary for tokenization, default `LatexVocab.default()`
        transform
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
    def __getitem__(self, _idx: int) -> Sample:
        pass

    @staticmethod
    def extract_features(ink: InkData) -> Features:
        """Extracts features of an InkData object

        Returns
        -------
        torch.Tensor of shape `(N, 12)` with features:
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
                Tan-acceleration `dspeed / dt`
            - is_stroke_start
                Binary indicators of stroke boundaries
            - y_center_rel, y_span_rel
                Vertical trace bounding-box features relative to expression bbox"""
        if ink.traces == []:
            return torch.zeros((0, _HMEDatasetBase.FEATURES), dtype=torch.float32)

        traces = [torch.as_tensor(t, dtype=torch.float32) for t in ink.traces if len(t) > 0]
        trace_lengths = [trace.shape[0] for trace in traces]

        all_points = torch.cat(traces, dim=0)
        all_points = _HMEDatasetBase._normalise_data(all_points)
        norm_traces = all_points.split(trace_lengths)

        expr_y_min = all_points[:, 1].min()
        expr_y_max = all_points[:, 1].max()
        expr_y_span = (expr_y_max - expr_y_min) + _HMEDatasetBase.EPS

        features_per_trace = []
        for trace in norm_traces:
            d_xyt = _HMEDatasetBase._trace_deltas(trace)
            dynamics = _HMEDatasetBase._trace_dynamics(d_xyt)
            trace_bbox = _HMEDatasetBase._trace_bbox_features(
                trace=trace,
                expr_y_min=expr_y_min,
                expr_y_span=expr_y_span,
            )

            is_stroke_start = torch.zeros(trace.shape[0], dtype=torch.float32, device=trace.device)
            is_stroke_start[0] = 1.0

            features_per_trace.append(
                torch.cat([trace, d_xyt, dynamics, is_stroke_start.unsqueeze(1), trace_bbox], dim=1)
            )

        feats = torch.cat(features_per_trace, dim=0)

        cols_to_norm = [3, 4, 5, 6, 7, 8]
        for col in cols_to_norm:
            mean = feats[:, col].mean()
            std = feats[:, col].std(unbiased=False)
            std = torch.nan_to_num(std, nan=0.0) + _HMEDatasetBase.EPS

            feats[:, col] = (feats[:, col] - mean) / std
            feats[:, col] = torch.clamp(feats[:, col], min=-5.0, max=5.0)

        feats = torch.nan_to_num(feats, nan=0.0, posinf=5.0, neginf=-5.0)

        return feats

    @staticmethod
    def _normalise_data(xyt: TensorF32) -> TensorF32:
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
    def _trace_deltas(xyt: TensorF32) -> TensorF32:
        """Compute `(dx, dy, dt)` for a single normalized trace.

        The first point in every trace has zero deltas by definition.
        """
        deltas = torch.zeros_like(xyt)
        if xyt.shape[0] > 1:
            deltas[1:] = xyt[1:] - xyt[:-1]
        return deltas

    @staticmethod
    def _trace_dynamics(d_xyt: TensorF32) -> TensorF32:
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
        if d_xyt.shape[0] == 1:
            return torch.zeros_like(d_xyt)

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

    @staticmethod
    def _trace_bbox_features(
        trace: TensorF32, expr_y_min: TensorF32, expr_y_span: TensorF32
    ) -> TensorF32:
        """Create vertical relative bounding-box features for a single trace.

        Parameters
        ----------
        trace : Tensor
            Normalized trace points with shape `(N_trace, 3)`.
        expr_y_min : Tensor
            Minimum y-coordinate for the full expression.
        expr_y_span : Tensor
            Full-expression vertical span (max - min)

        Returns
        -------
        Tensor
            Tensor `(N_trace, 2)` with columns `(y_center_rel, y_span_rel)`.
            `y_center_rel` describes vertical placement (top/middle/bottom),
            `y_span_rel` describes relative trace height.
        """
        y_coords = trace[:, 1]
        trace_y_min = y_coords.min()
        trace_y_max = y_coords.max()
        trace_y_center = 0.5 * (trace_y_min + trace_y_max)
        trace_y_span = trace_y_max - trace_y_min

        trace_bbox = torch.empty((trace.shape[0], 2), dtype=torch.float32, device=trace.device)
        trace_bbox[:, 0] = (trace_y_center - expr_y_min) / expr_y_span
        trace_bbox[:, 1] = trace_y_span / expr_y_span
        return trace_bbox


class HMEDatasetRaw(_HMEDatasetBase):
    """HME Dataset that operates on raw .inkml files"""

    def __init__(
        self,
        root: Path | str,
        split: str,
        vocab: LatexVocab | None = None,
        transform: Transformation[Features] | None = None,
    ):
        super().__init__(root=root, split=split, vocab=vocab, transform=transform)
        self.data_path = Path(root, split)
        self.filenames = sorted(self.data_path.rglob("*.inkml"))

    def __len__(self) -> int:
        return len(self.filenames)

    def __repr__(self) -> str:
        return f"HMEDataset(root={self.root!r}, split={self.split}, n_samples={len(self)})"

    def __getitem__(self, idx: int) -> Sample:
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
        transform: Transformation[Features] | None = None,
        min_len: int | None = None,
        max_len: int | None = None,
    ):
        super().__init__(root=root, split=split, vocab=vocab, transform=transform)
        data_path = Path(root, split + ".pt")

        logger.info(f"Loading preprocessed HMEDataset from: {data_path}")
        raw_data: list[Sample] = torch.load(data_path, weights_only=True)

        logger.info(f"Filtering data for split: {split}")
        self.data: list[Sample] = []
        short, long, has_inf, has_nan = 0, 0, 0, 0
        for fts, ts in raw_data:
            ft_len = fts.size(0)

            if min_len is not None and ft_len < min_len:
                short += 1

            elif max_len is not None and ft_len > max_len:
                long += 1

            elif torch.isinf(fts).any():
                has_inf += 1

            elif torch.isnan(fts).any():
                has_nan += 1

            else:
                self.data.append((fts, ts))

        kept, filtered = len(self.data), (short + long + has_inf + has_nan)

        if filtered:
            logger.warning(
                f"For min={min_len}, max={max_len} split={split} got short={short} | long={long} | has_inf={has_inf} | has_nan={has_nan} | total={kept}/{kept + filtered}"
            )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Sample:
        return self.data[idx]
