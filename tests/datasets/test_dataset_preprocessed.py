from __future__ import annotations

from pathlib import Path

import torch

from hand_to_tex.datasets.dataset import HMEDatasetPreprocessed


class TestHMEDatasetPreprocessed:
    def test_loads_all_samples_from_pt_file(self, preprocessed_pt_root: Path, vocab):
        ds = HMEDatasetPreprocessed(root=preprocessed_pt_root, split="train", vocab=vocab)
        assert len(ds) == 4

    def test_getitem_returns_features_and_tokens(self, preprocessed_pt_root: Path, vocab):
        ds = HMEDatasetPreprocessed(root=preprocessed_pt_root, split="train", vocab=vocab)
        features, tokens = ds[0]

        assert features.ndim == 2
        assert features.shape[1] == 10
        assert features.dtype == torch.float32
        assert tokens.dtype == torch.long
        assert tokens.numel() > 0

    def test_min_len_filters_short_samples(self, preprocessed_pt_root: Path, vocab):
        # Fixture lengths are 24, 28, 32, 24; min_len=30 keeps only the one of len 32.
        ds = HMEDatasetPreprocessed(
            root=preprocessed_pt_root, split="train", vocab=vocab, min_len=30
        )
        assert len(ds) == 1
        assert ds[0][0].shape[0] >= 30

    def test_max_len_filters_long_samples(self, preprocessed_pt_root: Path, vocab):
        # max_len=24 keeps only the two samples of length exactly 24.
        ds = HMEDatasetPreprocessed(
            root=preprocessed_pt_root, split="train", vocab=vocab, max_len=24
        )
        assert len(ds) == 2
        for i in range(len(ds)):
            assert ds[i][0].shape[0] <= 24

    def test_inf_and_nan_samples_are_filtered(self, tmp_path: Path, vocab):
        good = (torch.randn(10, 10, dtype=torch.float32),
                torch.tensor([vocab.SOS, vocab.EOS], dtype=torch.long))
        nan_sample = (torch.full((10, 10), float("nan"), dtype=torch.float32),
                      torch.tensor([vocab.SOS, vocab.EOS], dtype=torch.long))
        inf_sample = (torch.full((10, 10), float("inf"), dtype=torch.float32),
                      torch.tensor([vocab.SOS, vocab.EOS], dtype=torch.long))

        path = tmp_path / "train.pt"
        torch.save([good, nan_sample, inf_sample], path)

        ds = HMEDatasetPreprocessed(root=tmp_path, split="train", vocab=vocab)
        assert len(ds) == 1
