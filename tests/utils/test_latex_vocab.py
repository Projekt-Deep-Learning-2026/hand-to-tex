"""Tests for LatexVocab class."""

from hand_to_tex.utils import LatexVocab


class TestLatexVocabLoad:
    """Tests for LatexVocab.load() method."""

    def test_load_returns_instance(self, vocab: LatexVocab):
        """Loading vocab.json returns a LatexVocab object."""
        assert isinstance(vocab, LatexVocab)


class TestLatexVocabEncodeDecode:
    """Tests for encode/decode methods."""

    def test_encode_known_token(self, vocab: LatexVocab):
        """Known tokens return valid IDs."""
        assert vocab.encode('x') >= 0
        assert vocab.encode('\\alpha') >= 0

    def test_encode_unknown_returns_unknown_id(self, vocab: LatexVocab):
        """Unknown tokens return UNK token id."""
        assert vocab.encode('NONEXISTENT') == vocab.UNK

    def test_decode_invalid_id(self, vocab: LatexVocab):
        """Invalid IDs return '<UNK>' token."""
        assert vocab.decode(-1) == '<UNK>'

    def test_encode_decode_roundtrip(self, vocab: LatexVocab):
        """Encoding then decoding returns original token."""
        for token in ['x', '+', '\\frac', '\\mathbb{R}']:
            token_id = vocab.encode(token)
            assert vocab.decode(token_id) == token


class TestLatexVocabExpr:
    """Tests for expression tokenization."""

    def test_encode_expr_simple(self, vocab: LatexVocab):
        """Simple expression is tokenized correctly."""
        ids = vocab.encode_expr('x+y')
        assert len(ids) == 3

    def test_encode_expr_with_command(self, vocab: LatexVocab):
        """LaTeX commands are tokenized as single tokens."""
        ids = vocab.encode_expr('\\alpha')
        assert len(ids) == 1

    def test_encode_expr_complex(self, vocab: LatexVocab):
        """Complex expression tokenizes correctly."""
        ids = vocab.encode_expr('\\frac{a}{b}')
        # \frac, {, a, }, {, b, }
        assert len(ids) == 7

    def test_encode_expr_mathbb(self, vocab: LatexVocab):
        """Blackboard bold is tokenized as single token."""
        ids = vocab.encode_expr('\\mathbb{R}')
        assert len(ids) == 1

    def test_decode_sequence(self, vocab: LatexVocab):
        """Decode sequence returns list of tokens."""
        ids = vocab.encode_expr('x+y')
        tokens = vocab.decode_sequence(ids)
        assert tokens == ['x', '+', 'y']
