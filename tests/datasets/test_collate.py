"""Tests for HMECollateFunction.

Verifies that the collation logic correctly pads features and tokens,
handles batch sizes, and maintains correct data types.
"""

from __future__ import annotations

import torch

from hand_to_tex.datasets.collate import HMECollateFunction
from hand_to_tex.utils import LatexVocab


class TestHMECollateFunction:
    """Test suite for HMECollateFunction."""

    def test_pads_features_to_max_length(self, vocab: LatexVocab) -> None:
        """Features must be padded to the maximum length in the batch."""
        collate = HMECollateFunction(vocab)
        batch = [
            (torch.ones(2, 12), torch.tensor([1, 2], dtype=torch.long)),
            (2 * torch.ones(5, 12), torch.tensor([3], dtype=torch.long)),
        ]

        padded_ft, ft_lengths, padded_ts, ts_lengths = collate(batch)

        assert padded_ft.shape == (2, 5, 12)
        assert torch.allclose(ft_lengths, torch.tensor([2, 5]))
        # First sample padded with zeros at positions [2:5]
        assert torch.allclose(padded_ft[0, 2:], torch.zeros(3, 12))
        # Second sample has original values
        assert torch.allclose(padded_ft[1], 2 * torch.ones(5, 12))

    def test_pads_tokens_with_pad_idx(self, vocab: LatexVocab) -> None:
        """Tokens must be padded with the vocabulary's PAD index."""
        collate = HMECollateFunction(vocab)
        batch = [
            (torch.ones(2, 12), torch.tensor([1, 2, 3], dtype=torch.long)),
            (torch.ones(2, 12), torch.tensor([4], dtype=torch.long)),
        ]

        _, _, padded_ts, ts_lengths = collate(batch)

        assert padded_ts.shape == (2, 3)
        assert torch.allclose(ts_lengths, torch.tensor([3, 1]))
        # Second sample padded with PAD token
        assert padded_ts[1, 1].item() == vocab.PAD
        assert padded_ts[1, 2].item() == vocab.PAD

    def test_single_sample_batch_returns_correct_shapes(self, vocab: LatexVocab) -> None:
        """Batch with a single sample must still have correct output dimensions."""
        collate = HMECollateFunction(vocab)
        batch = [
            (torch.randn(7, 12), torch.tensor([5, 6, 7], dtype=torch.long)),
        ]

        padded_ft, ft_lengths, padded_ts, ts_lengths = collate(batch)

        assert padded_ft.shape == (1, 7, 12)
        assert padded_ts.shape == (1, 3)
        assert ft_lengths == torch.tensor([7])
        assert ts_lengths == torch.tensor([3])

    def test_equal_length_sequences_no_padding_needed(self, vocab: LatexVocab) -> None:
        """When all sequences have equal length, no padding should be added."""
        collate = HMECollateFunction(vocab)
        batch = [
            (torch.ones(3, 12), torch.tensor([1, 2], dtype=torch.long)),
            (2 * torch.ones(3, 12), torch.tensor([3, 4], dtype=torch.long)),
        ]

        padded_ft, ft_lengths, padded_ts, ts_lengths = collate(batch)

        assert padded_ft.shape == (2, 3, 12)
        assert padded_ts.shape == (2, 2)
        # Original values preserved
        assert torch.allclose(padded_ft[0], torch.ones(3, 12))
        assert torch.allclose(padded_ft[1], 2 * torch.ones(3, 12))

    def test_output_dtypes(self, vocab: LatexVocab) -> None:
        """Output tensors must have correct data types (float32 for features, long for tokens)."""
        collate = HMECollateFunction(vocab)
        batch = [
            (torch.ones(2, 12, dtype=torch.float32), torch.tensor([1], dtype=torch.long)),
        ]

        padded_ft, _, padded_ts, _ = collate(batch)

        assert padded_ft.dtype == torch.float32
        assert padded_ts.dtype == torch.long
