from pathlib import Path
from xml.etree import ElementTree

import torch
from torch import Tensor

from tex_gen.types import Features

type Trace = list[tuple[float, float, float]]
"""list of points in time (x, y, t)"""

TAG_PREFIX = r"{http://www.w3.org/2003/InkML}"
EPS = 1e-8


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
    normalised_fts = (
        2 * (fts - fts.min(dim=0).values) / (fts.max(dim=0).values - fts.min(dim=0).values + EPS)
        - 1
    )
    return normalised_fts
