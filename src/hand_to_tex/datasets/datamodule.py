from collections.abc import Callable
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
from torch import Tensor
from torch.utils.data import DataLoader

from hand_to_tex.datasets.dataloader import HMEDataLoaderFactory
from hand_to_tex.utils import LatexVocab


class HMELightningDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root: str = "data",
        vocab_path: Path | None = None,
        processed: bool = True,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        **kwargs: Any,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.root = Path(root)
        self.vocab = LatexVocab.default() if vocab_path is None else LatexVocab.load(vocab_path)

        self.processed = processed
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.data_kwargs = kwargs

        self.factory: HMEDataLoaderFactory | None = None

        self._train_dataloader: DataLoader | None = None
        self._val_dataloader: DataLoader | None = None
        self._test_dataloader: DataLoader | None = None

    def setup(self, stage: str | None = None):

        if self.factory is None:
            self.factory = HMEDataLoaderFactory(
                root=self.root,
                processed=self.processed,
                vocab=self.vocab,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )

        match stage:
            case "fit":
                self._train_dataloader = self.factory.custom(
                    split="train", transform=self._get_train_transform(), **self.data_kwargs
                )
                self._val_dataloader = self.factory.custom(
                    split="valid", transform=self._get_eval_transform(), **self.data_kwargs
                )
            case "test":
                self._test_dataloader = self.factory.custom(
                    split="test", transform=self._get_eval_transform(), **self.data_kwargs
                )
            case None:
                self._test_dataloader = self.factory.custom(
                    split="test", transform=self._get_eval_transform(), **self.data_kwargs
                )
                self._train_dataloader = self.factory.custom(
                    split="train", transform=self._get_train_transform(), **self.data_kwargs
                )
                self._val_dataloader = self.factory.custom(
                    split="valid", transform=self._get_eval_transform(), **self.data_kwargs
                )
            case "predict":
                pass

    def _get_train_transform(self) -> Callable[[Tensor], Tensor] | None:
        return None

    def _get_eval_transform(self) -> Callable[[Tensor], Tensor] | None:
        return None

    def train_dataloader(self) -> DataLoader:
        if self._train_dataloader is None:
            raise ValueError("Train dataloader should've been initialised by now")

        return self._train_dataloader

    def val_dataloader(self) -> DataLoader:
        if self._val_dataloader is None:
            raise ValueError("Validation dataloader should've been initialised by now")

        return self._val_dataloader

    def test_dataloader(self) -> DataLoader:
        if self._test_dataloader is None:
            raise ValueError("Testing dataloader factory should've been initialised by now")

        return self._test_dataloader

    def predict_dataloader(self):
        raise NotImplementedError
