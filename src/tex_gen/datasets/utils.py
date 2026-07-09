from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import torch
from matplotlib.axes import Axes
from torch import Tensor

from tex_gen.types import Features

type Trace = list[tuple[float, float, float]]
"""list of points in time (x, y, t)"""

TAG_PREFIX = r"{http://www.w3.org/2003/InkML}"
EPS = 1e-8
NUM_FEATURES = 4


def __trace_from_text(trace: str) -> Trace:
    points = []
    for point in trace.split(","):
        x, y, t = point.split(" ")
        points.append((float(x), float(y), float(t)))
    return points


def __trace_to_tensor(trace: Trace) -> Tensor:
    L = len(trace)

    raw_tensor = torch.tensor(trace, dtype=torch.float32)
    features = torch.empty((L, 4), dtype=torch.float32)

    # rewrite the x, y coordinates
    features[:, 0:2] = raw_tensor[:, 0:2]

    features[0, 2] = 0.0
    if L > 1:
        features[1:, 2] = raw_tensor[1:, 2] - raw_tensor[:-1, 2]
    features[:, 3] = -1.0
    features[0, 3] = 1.0

    return features


def __normalise_features(fts: Features):

    xy_min, xy_max = fts[:, :2].min(), fts[:, :2].max()
    dt_min, dt_max = fts[:, 2].min(), fts[:, 2].max()
    fts[:, :2] = 2 * (fts[:, :2] - xy_min) / (xy_max - xy_min + EPS) - 1
    fts[:, 2] = 2 * (fts[:, 2] - dt_min) / (dt_max - dt_min + EPS) - 1


def extract_features(pth: Path) -> Features:
    with open(file=pth, encoding="utf-8") as file:
        root = ElementTree.fromstring(file.read())

    traces: list[Trace] = []
    for elem in root:
        if elem.tag.removeprefix(TAG_PREFIX) == "trace":
            if (trace_text := elem.text) is not None:
                trace = __trace_from_text(trace=trace_text)
                if len(trace) > 0:
                    traces.append(trace)
    fts = torch.concat([__trace_to_tensor(trace=t) for t in traces])
    __normalise_features(fts=fts)
    return fts


def draw_features(fts: Features, ax: Axes) -> None:

    fts_np = fts.detach().cpu().numpy()

    x, y = fts_np[:, 0], fts_np[:, 1]
    is_new = fts_np[:, 3]

    starts = np.append(np.where(is_new > 0.0)[0], len(fts_np))
    for a, b in zip(starts[:-1], starts[1:], strict=True):
        ax.plot(x[a:b], y[a:b], color="black")

    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
