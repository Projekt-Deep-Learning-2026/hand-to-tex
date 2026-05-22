"""Integration tests for the data loading pipeline.

Verifies that raw InkML and preprocessed PT files correctly flow through
Datasets, CollateFunctions, and DataLoaders into correctly shaped batches.
"""

from __future__ import annotations

from pathlib import Path

import torch

from hand_to_tex.datasets import (
    HMECollateFunction,
    HMEDataLoaderFactory,
    HMEDatasetPreprocessed,
    HMEDatasetRaw,
    HMELightningDataModule,
)
from hand_to_tex.utils import LatexVocab


def _prepare_inkml_splits(tmp_path: Path, sample_inkml: Path, n_per_split: int = 2) -> Path:
    """Helper to create a temporary directory with InkML files organized by split."""
    root = tmp_path / "raw"
    content = sample_inkml.read_text(encoding="utf-8")
    for split in ("train", "valid", "test"):
        d = root / split
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n_per_split):
            (d / f"item_{i}.inkml").write_text(content, encoding="utf-8")
    return root


class TestDataLoadingPipelineRaw:
    """Integration tests for the raw (InkML) data loading path."""

    def test_inkml_to_batch_roundtrip(
        self, tmp_path: Path, sample_inkml: Path, vocab: LatexVocab
    ) -> None:
        """InkML files must be correctly loaded and collated into training batches."""
        root = _prepare_inkml_splits(tmp_path, sample_inkml, n_per_split=2)

        factory = HMEDataLoaderFactory(
            root=root,
            processed=False,
            vocab=vocab,
            batch_size=2,
            num_workers=0,
            pin_memory=False,
            min_len=None,
            max_len=None,
        )
        loader = factory.train()
        padded_ft, ft_lengths, padded_ts, ts_lengths = next(iter(loader))

        assert padded_ft.shape[0] == 2
        assert padded_ft.shape[2] == 12
        assert padded_ft.dtype == torch.float32
        assert padded_ts.dtype == torch.long
        assert ft_lengths.shape == (2,)
        assert ts_lengths.shape == (2,)
        assert (ft_lengths <= padded_ft.shape[1]).all()
        assert (ts_lengths <= padded_ts.shape[1]).all()

    def test_dataset_collate_compose_consistently(
        self, tmp_path: Path, sample_inkml: Path, vocab: LatexVocab
    ) -> None:
        """Datasets and CollateFunctions must work together to produce valid batches."""
        root = _prepare_inkml_splits(tmp_path, sample_inkml, n_per_split=3)
        ds = HMEDatasetRaw(root=root, split="train", vocab=vocab)
        collate = HMECollateFunction(vocab)
        batch = collate([ds[0], ds[1], ds[2]])

        assert len(batch) == 4
        padded_ft, _, padded_ts, _ = batch
        assert padded_ft.shape[0] == 3
        assert padded_ts.shape[0] == 3


class TestDataLoadingPipelinePreprocessed:
    """Integration tests for the preprocessed (.pt) data loading path."""

    def test_preprocessed_dataset_to_batch(
        self, preprocessed_pt_root: Path, vocab: LatexVocab
    ) -> None:
        """Preprocessed .pt files must be correctly loaded and collated into batches."""
        ds = HMEDatasetPreprocessed(root=preprocessed_pt_root, split="train", vocab=vocab)
        assert len(ds) > 0

        factory = HMEDataLoaderFactory(
            root=preprocessed_pt_root,
            processed=True,
            vocab=vocab,
            batch_size=2,
            num_workers=0,
            pin_memory=False,
            min_len=None,
            max_len=None,
        )
        loader = factory.train()
        padded_ft, ft_lengths, padded_ts, ts_lengths = next(iter(loader))

        assert padded_ft.dtype == torch.float32
        assert padded_ts.dtype == torch.long
        assert padded_ft.shape[0] == 2
        assert ft_lengths.shape == (2,)
        assert ts_lengths.shape == (2,)


class TestDataModuleEndToEnd:
    """End-to-end integration tests for the HMELightningDataModule."""

    def test_datamodule_setup_then_iterate(self, preprocessed_pt_root: Path) -> None:
        """The DataModule must successfully setup and provide iterable dataloaders."""
        dm = HMELightningDataModule(
            root=str(preprocessed_pt_root),
            processed=True,
            batch_size=2,
            num_workers=0,
            pin_memory=False,
        )
        dm.setup(stage="fit")

        train_batch = next(iter(dm.train_dataloader()))
        val_batch = next(iter(dm.val_dataloader()))

        for batch in (train_batch, val_batch):
            padded_ft, ft_lengths, padded_ts, ts_lengths = batch
            assert padded_ft.shape[2] == 12
            assert padded_ft.dtype == torch.float32
            assert padded_ts.dtype == torch.long
            assert ft_lengths.shape[0] == padded_ft.shape[0]
            assert ts_lengths.shape[0] == padded_ts.shape[0]
