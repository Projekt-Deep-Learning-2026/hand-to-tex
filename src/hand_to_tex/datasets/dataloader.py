from collections.abc import Callable
from pathlib import Path

from torch import Tensor
from torch.utils.data.dataloader import DataLoader

from hand_to_tex.datasets.collate import HMECollateFunction
from hand_to_tex.datasets.dataset import HMEDatasetRaw
from hand_to_tex.utils import LatexVocab


class HMEDataLoaderFactory:
    """Fabric class for initialization of HME dataset dataloaders

    Examples
    --------
    >>> from pathlib import Path
    >>> factory = HMEDataLoaderFactory(root="data/sample", batch_size=32)
    >>> train_loader = factory.train()
    >>> for batch in train_loader:
    ...     break  # Process batch
    """

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

        Returns
        -------
        DataLoader
            DataLoader for training data with shuffling and drop_last=True

        Examples
        --------
        >>> factory = HMEDataLoaderFactory(root="data/sample")
        >>> train_loader = factory.train()
        >>> for images, labels in train_loader:
        ...     # Process training batch
        ...     pass
        """
        dataset = HMEDatasetRaw(
            root=self.root, split="train", vocab=self.vocab, transform=transform
        )

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

        Examples
        --------
        >>> factory = HMEDataLoaderFactory(root="data/sample")
        >>> valid_loader = factory.valid()
        >>> for images, labels in valid_loader:
        ...     # Evaluate on validation batch
        ...     pass
        """
        dataset = HMEDatasetRaw(
            root=self.root, split="valid", vocab=self.vocab, transform=transform
        )

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

        Examples
        --------
        >>> factory = HMEDataLoaderFactory(root="data/sample")
        >>> test_loader = factory.test()
        >>> for images, labels in test_loader:
        ...     # Run inference on test batch
        ...     pass
        """
        dataset = HMEDatasetRaw(root=self.root, split="test", vocab=self.vocab, transform=transform)

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

        Examples
        --------
        >>> factory = HMEDataLoaderFactory(root="data/sample")
        >>> # Create custom dataloader with custom batch size
        >>> custom_loader = factory.custom("train", batch_size=64, num_workers=8)
        >>> # Create dataloader for custom split
        >>> custom_split_loader = factory.custom("custom_split")
        """
        split_name = split.lower()
        dataset = HMEDatasetRaw(
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
