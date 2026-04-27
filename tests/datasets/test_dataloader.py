from pathlib import Path

import torch
from torch.utils.data import RandomSampler, SequentialSampler

from hand_to_tex.datasets.dataloader import HMEDataLoaderFactory


def _prepare_splits_with_sample(tmp_path: Path, sample_inkml: Path) -> Path:
    root = tmp_path / "data"
    content = sample_inkml.read_text(encoding="utf-8")

    for split in ("train", "valid", "test"):
        split_dir = root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        (split_dir / f"{split}.inkml").write_text(content, encoding="utf-8")

    return root


class TestHMEDataLoaderFactory:
    def test_train_uses_shuffle_and_drop_last(self, tmp_path: Path, sample_inkml: Path, vocab):
        root = _prepare_splits_with_sample(tmp_path, sample_inkml)
        factory = HMEDataLoaderFactory(
            root=root,
            processed=False,
            vocab=vocab,
            batch_size=1,
            num_workers=0,
            pin_memory=False,
            min_len=None,
            max_len=None,
        )

        train_loader = factory.train()

        assert train_loader.drop_last is True
        assert isinstance(train_loader.sampler, RandomSampler)

    def test_valid_and_test_use_sequential_sampler(self, tmp_path: Path, sample_inkml: Path, vocab):
        root = _prepare_splits_with_sample(tmp_path, sample_inkml)
        factory = HMEDataLoaderFactory(
            root=root,
            vocab=vocab,
            processed=False,
            batch_size=1,
            num_workers=0,
            pin_memory=False,
            min_len=None,
            max_len=None,
        )

        valid_loader = factory.valid()
        test_loader = factory.test()

        assert valid_loader.drop_last is False
        assert test_loader.drop_last is False
        assert isinstance(valid_loader.sampler, SequentialSampler)
        assert isinstance(test_loader.sampler, SequentialSampler)

    def test_custom_overrides_defaults(self, tmp_path: Path, sample_inkml: Path, vocab):
        root = _prepare_splits_with_sample(tmp_path, sample_inkml)
        factory = HMEDataLoaderFactory(
            root=root,
            vocab=vocab,
            processed=False,
            batch_size=4,
            num_workers=0,
            pin_memory=False,
            min_len=None,
            max_len=None,
        )

        loader = factory.custom("valid", batch_size=2, shuffle=True, drop_last=True)

        assert loader.batch_size == 2
        assert loader.drop_last is True
        assert isinstance(loader.sampler, RandomSampler)

    def test_iteration_returns_collated_batch(self, tmp_path: Path, sample_inkml: Path, vocab):
        root = _prepare_splits_with_sample(tmp_path, sample_inkml)
        factory = HMEDataLoaderFactory(
            root=root,
            vocab=vocab,
            processed=False,
            batch_size=1,
            num_workers=0,
            pin_memory=False,
            min_len=None,
            max_len=None,
        )

        loader = factory.valid()
        padded_ft, ft_lengths, padded_ts, ts_lengths = next(iter(loader))

        assert padded_ft.ndim == 3
        assert padded_ft.shape[0] == 1  # batch size
        assert padded_ft.shape[2] == 12  # feature dim
        assert isinstance(ft_lengths, torch.Tensor)
        assert padded_ts.dtype == torch.long
        assert isinstance(ts_lengths, torch.Tensor)

    def test_transform_applied_in_loader(self, tmp_path: Path, sample_inkml: Path, vocab):
        root = _prepare_splits_with_sample(tmp_path, sample_inkml)
        factory = HMEDataLoaderFactory(
            root=root,
            vocab=vocab,
            processed=False,
            batch_size=1,
            num_workers=0,
            pin_memory=False,
            min_len=None,
            max_len=None,
        )

        def negate(x: torch.Tensor) -> torch.Tensor:
            return -x

        loader_transformed = factory.valid(transform=negate)
        loader_original = factory.valid()

        ft_transformed, _, _, _ = next(iter(loader_transformed))
        ft_original, _, _, _ = next(iter(loader_original))

        assert torch.allclose(ft_transformed, -ft_original)
