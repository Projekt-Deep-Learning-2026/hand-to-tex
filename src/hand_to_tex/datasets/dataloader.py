from collections.abc import Callable
from pathlib import Path

from torch import Tensor
from torch.utils.data.dataloader import DataLoader

from hand_to_tex.datasets.collate import HMECollateFunction
from hand_to_tex.datasets.dataset import HMEDatasetPreprocessed, HMEDatasetRaw
from hand_to_tex.utils import LatexVocab


class HMEDataLoaderFactory:
    """Fabric class for initialization of HME dataset dataloaders

    Examples
    --------
    >>> from pathlib import Path
    >>> factory = HMEDataLoaderFactory(root="data/sample", batch_size=32)
    >>> train_loader = factory.train()
    >>> valid_loader = factory.valid()
    >>> for batch in train_loader:
    ...     # Process batch
    """

    def __init__(
        self,
        root: Path | str,
        processed: bool,
        vocab: LatexVocab,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
    ):
        """Creates a DataLoader factory that can produce HMEDataset dataloaders

        Parameters
        ----------
        root : Path | str
            Path to the directory that contains splits like train, valid, test
        processed : bool
            Indicates whether data in `root` is `.inkml` (not processed) `.pt` (processed)
        vocab : LatexVocab | None
            Latex vocabulary that will be used, if None then `LatexVocab.default()` will be used
        batch_size : int
            batch_size passed to torch `DataLoader`
        num_workers : int
            num_workers passed to torch `DataLoader`
        pin_memory : bool
            pin_memory passed to torch `DataLoader`
        """
        self.root = Path(root)
        self.processed = processed
        self.vocab = vocab
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

        Returns
        -------
        DataLoader
            DataLoader for training data with shuffling and drop_last=True

        """
        dataset_type = HMEDatasetPreprocessed if self.processed else HMEDatasetRaw
        dataset = dataset_type(root=self.root, split="train", vocab=self.vocab, transform=transform)

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
            Transform method passed to HMEDatasetRaw

        Returns
        -------
        DataLoader
            DataLoader for validation data without shuffling and drop_last=False

        """
        dataset_type = HMEDatasetPreprocessed if self.processed else HMEDatasetRaw
        dataset = dataset_type(root=self.root, split="valid", vocab=self.vocab, transform=transform)

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
            Transform method passed to HMEDatasetRaw

        Returns
        -------
        DataLoader
            DataLoader for test data without shuffling and drop_last=False
        """
        dataset_type = HMEDatasetPreprocessed if self.processed else HMEDatasetRaw
        dataset = dataset_type(root=self.root, split="test", vocab=self.vocab, transform=transform)

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
            Transform method passed to HMEDatasetRaw
        **kwargs
            Additional keyword arguments forwarded to `torch.utils.data.DataLoader`

        Returns
        -------
        DataLoader
            Configured DataLoader for the specified split
        """
        split_name = split.lower()
        dataset_type = HMEDatasetPreprocessed if self.processed else HMEDatasetRaw
        dataset = dataset_type(
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
