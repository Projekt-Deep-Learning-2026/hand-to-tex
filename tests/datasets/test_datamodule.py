from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from hand_to_tex.datasets.datamodule import HMELightningDataModule


class TestDataModuleLifecycle:
    def test_setup_fit_creates_train_and_val_loaders(self, preprocessed_pt_root: Path):
        dm = HMELightningDataModule(
            root=str(preprocessed_pt_root),
            processed=True,
            batch_size=2,
            num_workers=0,
            pin_memory=False,
        )
        dm.setup(stage="fit")

        assert isinstance(dm.train_dataloader(), DataLoader)
        assert isinstance(dm.val_dataloader(), DataLoader)
        with pytest.raises(ValueError):
            dm.test_dataloader()

    def test_setup_test_creates_test_loader_only(self, preprocessed_pt_root: Path):
        dm = HMELightningDataModule(
            root=str(preprocessed_pt_root),
            processed=True,
            batch_size=2,
            num_workers=0,
            pin_memory=False,
        )
        dm.setup(stage="test")

        assert isinstance(dm.test_dataloader(), DataLoader)
        with pytest.raises(ValueError):
            dm.train_dataloader()

    def test_dataloader_yields_correctly_shaped_batches(self, preprocessed_pt_root: Path):
        dm = HMELightningDataModule(
            root=str(preprocessed_pt_root),
            processed=True,
            batch_size=2,
            num_workers=0,
            pin_memory=False,
        )
        dm.setup(stage="fit")
        batch = next(iter(dm.train_dataloader()))

        padded_ft, ft_lengths, padded_ts, ts_lengths = batch
        assert padded_ft.ndim == 3
        assert padded_ft.shape[2] == 12
        assert padded_ft.dtype == torch.float32
        assert ft_lengths.ndim == 1
        assert padded_ts.dtype == torch.long
        assert ts_lengths.ndim == 1

    def test_train_dataloader_raises_before_setup(self, preprocessed_pt_root: Path):
        dm = HMELightningDataModule(
            root=str(preprocessed_pt_root),
            processed=True,
            batch_size=2,
            num_workers=0,
            pin_memory=False,
        )
        with pytest.raises(ValueError):
            dm.train_dataloader()
        with pytest.raises(ValueError):
            dm.val_dataloader()
        with pytest.raises(ValueError):
            dm.test_dataloader()

    def test_predict_dataloader_is_not_implemented(self, preprocessed_pt_root: Path):
        dm = HMELightningDataModule(
            root=str(preprocessed_pt_root),
            processed=True,
            batch_size=2,
            num_workers=0,
            pin_memory=False,
        )
        with pytest.raises(NotImplementedError):
            dm.predict_dataloader()
