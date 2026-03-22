"""Tests for InkData class."""

from pathlib import Path
from xml.etree import ElementTree

from hand_to_tex.datasets.ink_data import InkData


class TestInkDataLoad:
    """Tests for InkData.load() method."""

    def test_load_returns_inkdata_instance(self, sample_inkml: Path):
        """Loading a valid .inkml file returns an InkData object."""
        result = InkData.load(sample_inkml)
        assert isinstance(result, InkData)

    def test_load_parses_sample_id(self, sample_inkml: Path):
        """Sample ID is correctly extracted from annotation."""
        result = InkData.load(sample_inkml)
        assert result.sample_id == "test_001"

    def test_load_parses_tag(self, sample_inkml: Path):
        """Split tag is correctly extracted."""
        result = InkData.load(sample_inkml)
        assert result.tag == "train"

    def test_load_parses_tex_labels(self, sample_inkml: Path):
        """LaTeX labels (raw and normalized) are extracted."""
        result = InkData.load(sample_inkml)
        assert result.tex_raw == "x + y"
        assert result.tex_norm == "x + y"

    def test_load_parses_traces(self, sample_inkml: Path):
        """Traces are parsed as list of point lists."""
        result = InkData.load(sample_inkml)
        assert len(result.traces) == 2
        assert len(result.traces[0]) == 3
        assert len(result.traces[1]) == 2

    def test_trace_point_format(self, sample_inkml: Path):
        """Each trace point is a tuple of (x, y, t) floats."""
        result = InkData.load(sample_inkml)
        point = result.traces[0][0]
        assert point == (10.0, 20.0, 0.0)
        assert all(isinstance(v, float) for v in point)


class TestInkDataSymbols:
    """Tests for symbols tag special handling."""

    def test_symbols_tag_copies_tex_raw_to_norm(
            self, minimal_symbols_inkml: Path):
        """For symbols tag, tex_norm is set equal to tex_raw."""
        result = InkData.load(minimal_symbols_inkml)
        assert result.tag == "symbols"
        assert result.tex_norm == result.tex_raw == "a"


class TestLoadAnnotation:
    """Tests for InkData._load_annotation() method."""

    def test_load_annotation_label(self):
        """Test parsing label annotation."""
        elem = ElementTree.fromstring(
            '<annotation type="label">test label</annotation>'
        )
        result = InkData._load_annotation(elem)
        assert result == {"tex_raw": "test label"}

    def test_load_annotation_sample_id(self):
        """Test parsing sampleId annotation."""
        elem = ElementTree.fromstring(
            '<annotation type="sampleId">sample_123</annotation>'
        )
        result = InkData._load_annotation(elem)
        assert result == {"sample_id": "sample_123"}

    def test_load_annotation_split_tag(self):
        """Test parsing splitTagOriginal annotation."""
        elem = ElementTree.fromstring(
            '<annotation type="splitTagOriginal">test</annotation>'
        )
        result = InkData._load_annotation(elem)
        assert result == {"tag": "test"}

    def test_load_annotation_normalized_label(self):
        """Test parsing normalizedLabel annotation."""
        elem = ElementTree.fromstring(
            '<annotation type="normalizedLabel">normalized</annotation>'
        )
        result = InkData._load_annotation(elem)
        assert result == {"tex_norm": "normalized"}

    def test_load_annotation_unsupported_type_returns_none(self):
        """Test that unsupported annotation types return None."""
        elem = ElementTree.fromstring(
            '<annotation type="unknown">value</annotation>'
        )
        result = InkData._load_annotation(elem)
        assert result is None

    def test_load_annotation_empty_text(self):
        """Test annotation with empty text returns empty string."""
        elem = ElementTree.fromstring('<annotation type="label"></annotation>')
        result = InkData._load_annotation(elem)
        assert result == {"tex_raw": ""}

    def test_load_annotation_no_text(self):
        """Test annotation with no text content returns empty string."""
        elem = ElementTree.fromstring('<annotation type="label"/>')
        result = InkData._load_annotation(elem)
        assert result == {"tex_raw": ""}


class TestLoadTrace:
    """Tests for InkData._load_trace() method."""

    def test_load_trace_single_point(self):
        """Test parsing trace with single point."""
        elem = ElementTree.fromstring("<trace>1.0 2.0 3.0</trace>")
        result = InkData._load_trace(elem)
        assert result == [(1.0, 2.0, 3.0)]

    def test_load_trace_multiple_points(self):
        """Test parsing trace with multiple points."""
        elem = ElementTree.fromstring(
            "<trace>1.0 2.0 0.0, 3.0 4.0 1.0, 5.0 6.0 2.0</trace>"
        )
        result = InkData._load_trace(elem)
        assert result == [(1.0, 2.0, 0.0), (3.0, 4.0, 1.0), (5.0, 6.0, 2.0)]

    def test_load_trace_no_text_returns_none(self):
        """Test that trace with no text content returns None."""
        elem = ElementTree.fromstring("<trace/>")
        result = InkData._load_trace(elem)
        assert result is None

    def test_load_trace_integer_values_converted_to_float(self):
        """Test that integer-looking values are converted to float."""
        elem = ElementTree.fromstring("<trace>1 2 3</trace>")
        result = InkData._load_trace(elem)
        assert result == [(1.0, 2.0, 3.0)]

    def test_load_trace_negative_values(self):
        """Test parsing trace with negative values."""
        elem = ElementTree.fromstring("<trace>-1.5 -2.5 0.0</trace>")
        result = InkData._load_trace(elem)
        assert result == [(-1.5, -2.5, 0.0)]
