from __future__ import annotations

from pathlib import Path

import pytest
import torch
from loguru import logger

from hand_to_tex.utils import LatexVocab


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_inkml(fixtures_dir: Path) -> Path:
    return fixtures_dir / "sample.inkml"


@pytest.fixture
def minimal_symbols_inkml(fixtures_dir: Path) -> Path:
    return fixtures_dir / "minimal_symbols.inkml"


@pytest.fixture
def caplog_loguru():
    """Capture loguru logs; exposes ``.text`` as a joined string."""
    handler_id = None
    messages: list[str] = []

    def capture_handler(message):
        messages.append(message.record["message"])

    handler_id = logger.add(capture_handler, format="{message}")

    class LogCapture:
        @property
        def text(self):
            return "\n".join(messages)

    yield LogCapture()
    if handler_id is not None:
        logger.remove(handler_id)


@pytest.fixture
def vocab() -> LatexVocab:
    return LatexVocab.default()


@pytest.fixture
def vocab_path() -> str:
    return str(Path("data/assets/vocab.json"))


@pytest.fixture
def tiny_model_kwargs(vocab_path: str) -> dict:
    return {
        "vocab_path": vocab_path,
        "max_generate_len": 8,
        "lr": 1e-3,
        "label_smoothing": 0.0,
        "weight_decay": 0.0,
    }


@pytest.fixture
def tiny_decoder_kwargs() -> dict:
    return {
        "d_model": 32,
        "nhead": 2,
        "num_encoder_layers": 1,
        "num_decoder_layers": 1,
        "dim_feedforward": 64,
        "dropout": 0.0,
    }


@pytest.fixture
def tiny_lit_module(tiny_model_kwargs: dict, tiny_decoder_kwargs: dict):
    from hand_to_tex.models.components import ExperimentalTransformer
    from hand_to_tex.models.lit_module import HMELightningModule

    torch.manual_seed(0)
    vocab = LatexVocab.load(tiny_model_kwargs["vocab_path"])
    model = ExperimentalTransformer(
        in_channels=12,
        vocab_size=len(vocab),
        pad_idx=vocab.PAD,
        d_model=tiny_decoder_kwargs["d_model"],
        nhead=tiny_decoder_kwargs["nhead"],
        num_encoder_layers=tiny_decoder_kwargs["num_encoder_layers"],
        num_decoder_layers=tiny_decoder_kwargs["num_decoder_layers"],
        dim_feedforward=tiny_decoder_kwargs["dim_feedforward"],
        dropout=tiny_decoder_kwargs["dropout"],
    )
    return HMELightningModule(model=model, **tiny_model_kwargs)


@pytest.fixture
def synthetic_batch(vocab: LatexVocab):
    """Deterministic 4-tuple matching HMECollateFunction output."""
    torch.manual_seed(0)
    B, T_src, F = 2, 24, 12
    padded_features = torch.randn(B, T_src, F, dtype=torch.float32)
    feature_lengths = torch.tensor([T_src, T_src - 4], dtype=torch.long)

    sos, eos, pad = vocab.SOS, vocab.EOS, vocab.PAD
    a = vocab.encode("x")
    b = vocab.encode("+")

    padded_tokens = torch.tensor(
        [
            [sos, a, b, a, eos],
            [sos, a, eos, pad, pad],
        ],
        dtype=torch.long,
    )
    token_lengths = torch.tensor([5, 3], dtype=torch.long)

    return padded_features, feature_lengths, padded_tokens, token_lengths


def _make_pt_split(path: Path, n_samples: int, vocab: LatexVocab) -> None:
    samples = []
    for i in range(n_samples):
        # Vary length so collation actually pads across samples.
        T = 24 + (i % 3) * 4
        features = torch.randn(T, 12, dtype=torch.float32)
        tokens = torch.tensor(
            [vocab.SOS, vocab.encode("x"), vocab.encode("+"), vocab.encode("y"), vocab.EOS],
            dtype=torch.long,
        )
        samples.append((features, tokens))
    torch.save(samples, path)


@pytest.fixture
def preprocessed_pt_root(tmp_path: Path, vocab: LatexVocab) -> Path:
    root = tmp_path / "processed"
    root.mkdir(parents=True, exist_ok=True)
    for split in ("train", "valid", "test"):
        _make_pt_split(root / f"{split}.pt", n_samples=4, vocab=vocab)
    return root
