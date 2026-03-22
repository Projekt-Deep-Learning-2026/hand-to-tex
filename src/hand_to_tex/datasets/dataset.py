from pathlib import Path
import torch
from torch.utils.data.dataset import Dataset
from typing import Optional, Callable

from .ink_data import InkData
from ..utils import LatexVocab


class HMEDataset(Dataset):
    """Handwritten Mathematical Expressions dataset for processing
    `.inkml` files
    """
    def __init__(
                self,
                root: Path | str,
                vocab: Optional[LatexVocab] = None,
                transform: Optional[Callable] = None
            ):
        """Creates the dataset

        Parameters
        ----------

        root : Path | str
            Directory of all dataset files
        vocab : LatexVocab | None
            Vocabulary for tokenization, default `LatexVocab.default()`
        """

        self.root = Path(root)
        self.filenames = sorted(list(self.root.rglob('*.inkml')))

        self.vocab = LatexVocab.default() if vocab is None else vocab
        self.transform = transform

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int):
        raise NotImplementedError


if __name__ == "__main__":
    d = HMEDataset('data/train')
    for x in d.filenames:
        print(Path(d.root, x))
