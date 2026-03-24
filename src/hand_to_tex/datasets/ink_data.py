from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias
from xml.etree import ElementTree

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes


# Single trace point stored as (x, y, t).
TracePoint: TypeAlias = tuple[float, float, float]
# Collection of traces: each trace is a list of points.
Traces:     TypeAlias = list[list[TracePoint]]


@dataclass
class InkData:
    """Representation of a single InkML sample.

    Stores metadata and stroke data parsed from an InkML file.

    Fields
    ------
    - tag : Literal['train', 'test', 'valid', 'symbols', 'synthetic']
        Data split identifier.
    - sample_id : str
        Unique sample identifier.
    - tex_raw : str
        Raw LaTeX label.
    - tex_norm : str
        Normalized LaTeX label.
    - traces : Traces
        List of strokes; each stroke is a list of `(x, y, t)` points.

    Examples
    --------
    >>> data = InkData.load("sample.inkml")
    >>> data.sample_id
    'sample_001'
    >>> fig, ax = data.to_fig()
    """
    tag:       Literal['train', 'test', 'valid', 'symbols', 'synthetic']
    sample_id: str
    tex_raw:   str
    tex_norm:  str
    traces:    Traces

    ENCODING = 'utf-8'
    TAG_PREFIX = r"{http://www.w3.org/2003/InkML}"
    ANNOTATION_MAP = {
        'label':            'tex_raw',
        'splitTagOriginal': 'tag',
        'sampleId':         'sample_id',
        'normalizedLabel':  'tex_norm'
    }

    @staticmethod
    def load(path: Path | str) -> InkData:
        """Load an InkML file and build an `InkData` object.

        Parameters
        ----------
        path : Path | str
            Path to the `.inkml` file.

        Returns
        -------
        InkData
            Parsed sample with metadata and a list of strokes.
        """

        with open(path, 'r', encoding=InkData.ENCODING) as file:
            root = ElementTree.fromstring(file.read())

        kwargs = {}
        traces = []
        for elem in root:
            match elem.tag.removeprefix(InkData.TAG_PREFIX):
                case 'annotation':
                    if (kwarg := InkData._load_annotation(elem)) is not None:
                        kwargs.update(kwarg)
                case 'trace':
                    if (trace := InkData._load_trace(elem)) is not None:
                        traces.append(trace)

        if kwargs.get('tag') == 'symbols':
            kwargs['tex_norm'] = kwargs['tex_raw']

        return InkData(traces=traces, **kwargs)

    @staticmethod
    def _load_annotation(elem: ElementTree.Element) -> dict[str, str] | None:
        """Map an `annotation` element to a supported `InkData` field.

        Parameters
        ----------
        elem : ElementTree.Element
            XML `annotation` element.

        Returns
        -------
        dict[str, str] | None
            Dictionary `{field_name: value}` or `None`
            if the annotation type is not supported.
        """
        t = str(elem.attrib.get('type'))
        if (k := InkData.ANNOTATION_MAP.get(t)):
            v = elem.text
            return {k: ('' if v is None else v)}
        else:
            return None

    @staticmethod
    def _load_trace(elem: ElementTree.Element) -> list[TracePoint] | None:
        """Parse a `trace` element into a list of `(x, y, t)` points.

        Parameters
        ----------
        elem : ElementTree.Element
            XML `trace` element.

        Returns
        -------
        list[TracePoint] | None
            Trace points list or `None` if the trace has no content.
        """

        if (txt := elem.text) is None:
            return None
        trace = []
        for point in txt.split(','):
            x, y, t = map(float, point.split())
            trace.append((x, y, t))

        return None if trace == [] else trace

    def to_fig(
            self,
            *,
            figsize: tuple[int, int] = (6, 4),
            linewidth:         float = 2.0,
            color:               str = 'black',
            invert_y:           bool = True,
            **kwargs
            ) -> tuple[Figure, Axes]:
        """Render sample traces using matplotlib.

        Parameters
        ----------
        figsize : tuple[int, int], default=(6, 4)
            Figure size passed to matplotlib.
        linewidth : float, default=2.0
            Width of the plotted stroke lines.
        color : str, default='black'
            Stroke color used for all traces.
        invert_y : bool, default=True
            Whether to invert Y axis for handwriting-like view.
        **kwargs
            Extra keyword arguments forwarded to `plt.figure`.

        Returns
        -------
        tuple[matplotlib.Figure, matplotlib.Axes]
            Created matplotlib figure and axes. test
        """
        fig = plt.figure(num=self.sample_id, figsize=figsize, **kwargs)
        ax = fig.add_subplot(1, 1, 1)

        for trace in self.traces:
            if not trace:
                continue
            x_coords = [point[0] for point in trace]
            y_coords = [point[1] for point in trace]
            ax.plot(x_coords, y_coords, color=color, linewidth=linewidth)

        ax.set_aspect('equal', adjustable='box')
        ax.axis('off')

        title = f"Sample: {self.sample_id} ( {self.tag} )\n"
        title += rf"' ${self.tex_norm}$ '"

        ax.set_title(title)

        if invert_y:
            ax.invert_yaxis()

        return fig, ax
