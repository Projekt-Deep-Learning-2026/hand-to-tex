from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import torch
from matplotlib.axes import Axes

from tex_gen.types import Features

type Trace = list[tuple[float, float, float]]
"""list of points in time (x, y, t)"""

TAG_PREFIX = r"{http://www.w3.org/2003/InkML}"
EPS = 1e-8
NUM_FEATURES = 4


def __trace_from_text(trace: str) -> Trace:
    points = []
    for point in trace.split(","):
        x, y, t = point.split()
        points.append((float(x), float(y), float(t)))
    return points


def __normalise_features(fts: Features):
    xy_max = fts[:, :2].abs().max()
    fts[:, :2] = fts[:, :2] / (xy_max + EPS)

    dt_min, dt_max = fts[:, 2].min(), fts[:, 2].max()
    fts[:, 2] = 2 * (fts[:, 2] - dt_min) / (dt_max - dt_min + EPS) - 1


def extract_features(pth: Path) -> Features:
    with open(file=pth, encoding="utf-8") as file:
        root = ElementTree.fromstring(file.read())

    all_points = []
    is_new = []

    for elem in root:
        if elem.tag.removeprefix(TAG_PREFIX) == "trace":
            if (trace_text := elem.text) is not None:
                trace = __trace_from_text(trace=trace_text)
                if len(trace) > 0:
                    for i, pt in enumerate(trace):
                        all_points.append(pt)
                        is_new.append(1.0 if i == 0 else -1.0)

    if len(all_points) == 0:
        return torch.empty((0, 4), dtype=torch.float32)

    raw_tensor = torch.tensor(all_points, dtype=torch.float32)
    is_new_tensor = torch.tensor(is_new, dtype=torch.float32)

    features = torch.empty((len(all_points), 4), dtype=torch.float32)

    # First point dx, dy, dt = 0
    features[0, 0:3] = 0.0

    if len(all_points) > 1:
        features[1:, 0:3] = raw_tensor[1:, 0:3] - raw_tensor[:-1, 0:3]

    features[:, 3] = is_new_tensor

    __normalise_features(fts=features)
    return features


def draw_features(fts: Features, ax: Axes) -> None:

    fts_np = fts.detach().cpu().numpy()

    # Features are now (dx, dy, dt, is_new), so cumsum recovers absolute positions
    dx, dy = fts_np[:, 0], fts_np[:, 1]
    x = np.cumsum(dx)
    y = np.cumsum(dy)

    is_new = fts_np[:, 3]

    starts = np.append(np.where(is_new > 0.0)[0], len(fts_np))
    for a, b in zip(starts[:-1], starts[1:], strict=True):
        ax.plot(x[a:b], y[a:b], color="black")

    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
