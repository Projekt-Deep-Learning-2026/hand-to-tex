from pathlib import Path

import torch

from hand_to_tex.datasets.dataset import HMEDatasetRaw
from hand_to_tex.datasets.ink_data import InkData


def _prepare_split_with_sample(tmp_path: Path, split: str, sample_inkml: Path) -> Path:
    root = tmp_path / "data"
    split_dir = root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / "sample.inkml").write_text(
        sample_inkml.read_text(encoding="utf-8"), encoding="utf-8"
    )
    return root


class TestHMEDataset:
    def test_len_returns_number_of_inkml_files(self, tmp_path: Path, sample_inkml: Path, vocab):
        root = _prepare_split_with_sample(tmp_path, "train", sample_inkml)
        (root / "train" / "second.inkml").write_text(
            sample_inkml.read_text(encoding="utf-8"), encoding="utf-8"
        )

        dataset = HMEDatasetRaw(root=root, split="train", vocab=vocab)

        assert len(dataset) == 2

    def test_getitem_returns_feature_tensor_and_long_tokens(
        self, tmp_path: Path, sample_inkml: Path, vocab
    ):
        root = _prepare_split_with_sample(tmp_path, "train", sample_inkml)

        dataset = HMEDatasetRaw(root=root, split="train", vocab=vocab)
        features, tokens = dataset[0]

        assert features.ndim == 2
        assert features.shape[1] == 12
        assert features.dtype == torch.float32
        assert tokens.dtype == torch.long
        assert tokens.numel() > 0

    def test_transform_is_applied_to_features(self, tmp_path: Path, sample_inkml: Path, vocab):
        root = _prepare_split_with_sample(tmp_path, "train", sample_inkml)

        def double_transform(x: torch.Tensor) -> torch.Tensor:
            return x * 2

        dataset = HMEDatasetRaw(root=root, split="train", vocab=vocab, transform=double_transform)
        features_transformed, _ = dataset[0]

        dataset_no_transform = HMEDatasetRaw(root=root, split="train", vocab=vocab)
        features_original, _ = dataset_no_transform[0]

        assert torch.allclose(features_transformed, features_original * 2)


class TestExtractFeatures:
    def test_empty_traces_returns_empty_tensor(self):
        ink = InkData(
            tag="train",
            sample_id="empty",
            tex_raw="",
            tex_norm="",
            traces=[],
        )

        features = HMEDatasetRaw.extract_features(ink)

        assert features.shape == (0, 12)
        assert features.dtype == torch.float32

    def test_single_point_trace_has_zero_deltas_and_dynamics(self):
        ink = InkData(
            tag="train",
            sample_id="single",
            tex_raw="x",
            tex_norm="x",
            traces=[[(5.0, 5.0, 0.0)]],
        )

        features = HMEDatasetRaw.extract_features(ink)
        assert features.shape == (1, 12)
        assert torch.equal(features[0, 3:9], torch.zeros(6))
        assert features[0, 9] == 1.0

    def test_multiple_traces_mark_stroke_starts_correctly(self):
        ink = InkData(
            tag="train",
            sample_id="multi",
            tex_raw="xy",
            tex_norm="xy",
            traces=[
                [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)],
                [(2.0, 2.0, 2.0), (3.0, 3.0, 3.0), (4.0, 4.0, 4.0)],
            ],
        )

        features = HMEDatasetRaw.extract_features(ink)

        assert features.shape == (5, 12)
        is_stroke_start = features[:, 9]
        expected = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0])
        assert torch.equal(is_stroke_start, expected)

    def test_normalisation_preserves_aspect_ratio(self):
        # Rectangle wider than tall: x in [0,10], y in [0,5]
        ink = InkData(
            tag="train",
            sample_id="rect",
            tex_raw="r",
            tex_norm="r",
            traces=[
                [(0.0, 0.0, 0.0), (10.0, 5.0, 1.0)],
            ],
        )

        features = HMEDatasetRaw.extract_features(ink)

        x_norm = features[:, 0]
        y_norm = features[:, 1]
        # x range should be [0, 1], y range should be [0, 0.5] (aspect preserved)
        assert torch.isclose(x_norm[0], torch.tensor(0.0))
        assert torch.isclose(x_norm[1], torch.tensor(1.0), atol=1e-5)
        assert torch.isclose(y_norm[0], torch.tensor(0.0))
        assert torch.isclose(y_norm[1], torch.tensor(0.5), atol=1e-5)
