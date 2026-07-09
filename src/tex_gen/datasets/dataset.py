from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from torch.utils.data import Dataset

from tex_gen.datasets.utils import extract_features
from tex_gen.types import Features

type Transform = Callable[[Path], Features]


class TexGenDataset(Dataset[Features]):
    def __init__(self, folder_path: Path | str, transform: Transform = extract_features):
        self.root = Path(folder_path)
        self.transform = transform

        self.files: list[Path] = sorted(self.root.glob("*inkml"))

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Features:
        file_path = self.files[index]

        fts = self.transform(file_path)
        return fts
