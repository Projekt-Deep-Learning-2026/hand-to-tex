"""Tests for the InkData class and InkML parsing.

Validates that InkML files are correctly parsed into InkData instances,
handling various annotation types, trace formats, and symbols tag special cases.
"""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree

from hand_to_tex.datasets.ink_data import InkData


class TestInkDataLoad:
    """Test suite for the InkData.load() method."""

    def test_load_returns_inkdata_instance(self, sample_inkml: Path) -> None:
        """Loading a valid .inkml file must return an InkData instance."""
        result = InkData.load(sample_inkml)
        assert isinstance(result, InkData)

    def test_load_parses_sample_id(self, sample_inkml: Path) -> None:
        """The sample ID must be correctly extracted from the InkML annotation."""
        result = InkData.load(sample_inkml)
        assert result.sample_id == "test_001"

    def test_load_parses_tag(self, sample_inkml: Path) -> None:
        """The dataset split tag must be correctly extracted from the InkML annotation."""
        result = InkData.load(sample_inkml)
        assert result.tag == "train"

    def test_load_parses_tex_labels(self, sample_inkml: Path) -> None:
        """Both raw and normalized LaTeX labels must be extracted correctly."""
        result = InkData.load(sample_inkml)
        assert result.tex_raw == "x + y"
        assert result.tex_norm == "x + y"

    def test_load_parses_traces(self, sample_inkml: Path) -> None:
        """Pen traces must be parsed as a list of point lists with correct counts."""
        result = InkData.load(sample_inkml)
        assert len(result.traces) == 2
        assert len(result.traces[0]) == 3
        assert len(result.traces[1]) == 2

    def test_trace_point_format(self, sample_inkml: Path) -> None:
        """Each trace point must be a tuple of (x, y, t) floats."""
        result = InkData.load(sample_inkml)
        point = result.traces[0][0]
        assert point == (10.0, 20.0, 0.0)
        assert all(isinstance(v, float) for v in point)


class TestInkDataSymbols:
    """Test suite for specialized symbols tag handling."""

    def test_symbols_tag_copies_tex_raw_to_norm(self, minimal_symbols_inkml: Path) -> None:
        """For 'symbols' tagged files, tex_norm should be automatically set to tex_raw."""
        result = InkData.load(minimal_symbols_inkml)
        assert result.tag == "symbols"
        assert result.tex_norm == result.tex_raw == "a"


class TestLoadAnnotation:
    """Test suite for the internal InkData._load_annotation() method."""

    def test_load_annotation_label(self) -> None:
        """Parsing 'label' annotation must return the correct dict entry."""
        elem = ElementTree.fromstring('<annotation type="label">test label</annotation>')
        result = InkData._load_annotation(elem)
        assert result == {"tex_raw": "test label"}

    def test_load_annotation_sample_id(self) -> None:
        """Parsing 'sampleId' annotation must return the correct dict entry."""
        elem = ElementTree.fromstring('<annotation type="sampleId">sample_123</annotation>')
        result = InkData._load_annotation(elem)
        assert result == {"sample_id": "sample_123"}

    def test_load_annotation_split_tag(self) -> None:
        """Parsing 'splitTagOriginal' annotation must return the correct dict entry."""
        elem = ElementTree.fromstring('<annotation type="splitTagOriginal">test</annotation>')
        result = InkData._load_annotation(elem)
        assert result == {"tag": "test"}

    def test_load_annotation_normalized_label(self) -> None:
        """Parsing 'normalizedLabel' annotation must return the correct dict entry."""
        elem = ElementTree.fromstring('<annotation type="normalizedLabel">normalized</annotation>')
        result = InkData._load_annotation(elem)
        assert result == {"tex_norm": "normalized"}

    def test_load_annotation_unsupported_type_returns_none(self) -> None:
        """Unsupported annotation types must be ignored and return None."""
        elem = ElementTree.fromstring('<annotation type="unknown">value</annotation>')
        result = InkData._load_annotation(elem)
        assert result is None

    def test_load_annotation_empty_text(self) -> None:
        """Annotations with empty text should return an empty string for the value."""
        elem = ElementTree.fromstring('<annotation type="label"></annotation>')
        result = InkData._load_annotation(elem)
        assert result == {"tex_raw": ""}

    def test_load_annotation_no_text(self) -> None:
        """Annotations with no text content at all should return an empty string."""
        elem = ElementTree.fromstring('<annotation type="label"/>')
        result = InkData._load_annotation(elem)
        assert result == {"tex_raw": ""}


class TestLoadTrace:
    """Test suite for the internal InkData._load_trace() method."""

    def test_load_trace_single_point(self) -> None:
        """Parsing a trace with a single point must yield a one-item list of floats."""
        elem = ElementTree.fromstring("<trace>1.0 2.0 3.0</trace>")
        result = InkData._load_trace(elem)
        assert result == [(1.0, 2.0, 3.0)]

    def test_load_trace_multiple_points(self) -> None:
        """Parsing a trace with multiple points must yield the correct list of tuples."""
        elem = ElementTree.fromstring("<trace>1.0 2.0 0.0, 3.0 4.0 1.0, 5.0 6.0 2.0</trace>")
        result = InkData._load_trace(elem)
        assert result == [(1.0, 2.0, 0.0), (3.0, 4.0, 1.0), (5.0, 6.0, 2.0)]

    def test_load_trace_no_text_returns_none(self) -> None:
        """Traces with no text content must return an empty list."""
        elem = ElementTree.fromstring("<trace/>")
        result = InkData._load_trace(elem)
        assert result == []

    def test_load_trace_integer_values_converted_to_float(self) -> None:
        """Point coordinates provided as integers must be automatically converted to floats."""
        elem = ElementTree.fromstring("<trace>1 2 3</trace>")
        result = InkData._load_trace(elem)
        assert result == [(1.0, 2.0, 3.0)]

    def test_load_trace_negative_values(self) -> None:
        """Traces containing negative coordinates must be parsed correctly."""
        elem = ElementTree.fromstring("<trace>-1.5 -2.5 0.0</trace>")
        result = InkData._load_trace(elem)
        assert result == [(-1.5, -2.5, 0.0)]
