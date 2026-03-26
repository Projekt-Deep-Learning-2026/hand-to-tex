from collections.abc import Callable
from pathlib import Path

from torch import Tensor
from torch.utils.data.dataloader import DataLoader

from hand_to_tex.datasets.collate import HMECollateFunction
from hand_to_tex.datasets.dataset import HMEDataset
from hand_to_tex.utils import LatexVocab


class HMEDataLoaderFactory:
    """Fabric class for initialization of HME dataset dataloaders"""

    def __init__(
        self,
        root: Path | str,
        vocab: LatexVocab | None = None,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        self.root = Path(root)
        self.vocab = LatexVocab.default() if vocab is None else vocab
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.collate_fn = HMECollateFunction(self.vocab)

    def train(self, transform: Callable[[Tensor], Tensor] | None = None) -> DataLoader:
        """Creates dataloader for `train` split

        Parameters
        ----------
        transform : (Callable: `Tensor` -> `Tensor`) | None
            Transform method passed to HMEDataset
        """
        dataset = HMEDataset(root=self.root, split="train", vocab=self.vocab, transform=transform)

        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=self.pin_memory,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

    def valid(self, transform: Callable[[Tensor], Tensor] | None = None) -> DataLoader:
        """Creates dataloader for `valid` split

        Parameters
        ----------
        transform : (Callable: `Tensor` -> `Tensor`) | None
            Transform method passed to HMEDataset
        """
        dataset = HMEDataset(root=self.root, split="valid", vocab=self.vocab, transform=transform)

        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=self.pin_memory,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

    def test(self, transform: Callable[[Tensor], Tensor] | None = None) -> DataLoader:
        """Creates dataloader for `test` split

        Parameters
        ----------
        transform : (Callable: `Tensor` -> `Tensor`) | None
            Transform method passed to HMEDataset
        """
        dataset = HMEDataset(root=self.root, split="test", vocab=self.vocab, transform=transform)

        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=self.pin_memory,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

    def custom(
        self,
        split: str,
        transform: Callable[[Tensor], Tensor] | None = None,
        **kwargs,
    ) -> DataLoader:
        """Creates dataloader for custom split and DataLoader kwargs.

        Parameters
        ----------
        split : str
            Dataset split name (e.g. `train`, `valid`, `test`)
        transform : (Callable: `Tensor` -> `Tensor`) | None
            Transform method passed to HMEDataset
        **kwargs
            Additional keyword arguments forwarded to `torch.utils.data.DataLoader`
        """
        split_name = split.lower()
        dataset = HMEDataset(
            root=self.root, split=split_name, vocab=self.vocab, transform=transform
        )

        defaults = {
            "dataset": dataset,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "collate_fn": self.collate_fn,
            "shuffle": split_name == "train",
            "drop_last": split_name == "train",
        }
        defaults.update(kwargs)

        return DataLoader(**defaults)
