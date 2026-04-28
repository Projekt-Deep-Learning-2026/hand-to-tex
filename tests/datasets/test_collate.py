import torch

from hand_to_tex.datasets.collate import HMECollateFunction


class TestHMECollateFunction:
    def test_pads_features_to_max_length(self, vocab):
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

    def test_pads_tokens_with_pad_idx(self, vocab):
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

    def test_single_sample_batch_returns_correct_shapes(self, vocab):
        collate = HMECollateFunction(vocab)
        batch = [
            (torch.randn(7, 12), torch.tensor([5, 6, 7], dtype=torch.long)),
        ]

        padded_ft, ft_lengths, padded_ts, ts_lengths = collate(batch)

        assert padded_ft.shape == (1, 7, 12)
        assert padded_ts.shape == (1, 3)
        assert ft_lengths == torch.tensor([7])
        assert ts_lengths == torch.tensor([3])

    def test_equal_length_sequences_no_padding_needed(self, vocab):
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

    def test_output_dtypes(self, vocab):
        collate = HMECollateFunction(vocab)
        batch = [
            (torch.ones(2, 12, dtype=torch.float32), torch.tensor([1], dtype=torch.long)),
        ]

        padded_ft, _, padded_ts, _ = collate(batch)

        assert padded_ft.dtype == torch.float32
        assert padded_ts.dtype == torch.long
