from pathlib import Path

import lightning.pytorch as pl
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Subset, random_split

from tex_gen.datasets.dataset import TexGenDataset
from tex_gen.types import Batch, Features


class TexGenDataModule(pl.LightningDataModule):
    def __init__(
        self, data_dir: Path | str, batch_size: int, num_workers: int, val_split: float
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split

        self.train_dataset: TexGenDataset | Subset[Features] | None = None
        self.val_dataset: TexGenDataset | Subset[Features] | None = None

    def setup(self, stage: str) -> None:
        if self.train_dataset is not None and self.val_dataset is not None:
            return

        full_dataset = TexGenDataset(folder_path=self.data_dir)

        self.train_dataset, self.val_dataset = random_split(
            full_dataset, lengths=[(1.0 - self.val_split), self.val_split]
        )

    @staticmethod
    def collate_fn(batch: list[Features]) -> Batch:

        padded_fts = pad_sequence(sequences=batch, batch_first=True)
        L = padded_fts.size(1)
        B = padded_fts.size(0)
        lengths = torch.tensor([x.size(0) for x in batch], dtype=torch.long)

        positions = torch.arange(L).unsqueeze(0).expand(B, L)
        mask = positions < lengths.unsqueeze(1)

        return padded_fts, mask

    def train_dataloader(self) -> DataLoader:
        assert self.train_dataset, "Train dataset not initialised in generative datamodule"
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_dataset, "Validation dataset not initialised in generative datamodule"
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )
