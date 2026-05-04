"""Benchmark: autoregressive inference with vs. without KV-Cache.

Loads the real test set from ``data/mathwriting-2024`` and a trained checkpoint
from ``data/last.ckpt``, then compares generation speed between:
    • **baseline** (lit_module)          — re-decodes the full prefix at every step
    • **kv-cache** (lit_module)          — caches K/V projections, single token per step

Usage
-----
    uv run htt-benchmark                             # CPU (default)
    uv run htt-benchmark --device mps                # Apple Silicon
    uv run htt-benchmark --device cuda               # NVIDIA GPU
    uv run htt-benchmark --num-batches 5             # limit batches
    uv run htt-benchmark --ckpt data/last.ckpt       # custom checkpoint
"""

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from hand_to_tex.datasets import HMEDataLoaderFactory
from hand_to_tex.models.components import (
    ExperimentalTransformer,
    ExperimentalTransformerKVCache,
)
from hand_to_tex.models.lit_module import HMELightningModule
from hand_to_tex.utils import LatexVocab

ROOT = Path("data/mathwriting-2024")
VOCAB_PATH = Path("data/assets/vocab.json")
CKPT_PATH = Path("data/last.ckpt")

MAX_GENERATE_LEN = 150
BATCH_SIZE = 256


@dataclass
class RunStats:
    """Accumulates per-batch timings for a single model variant."""

    label: str
    times_ms: list[float] = field(default_factory=list)

    @property
    def total_ms(self) -> float:
        return sum(self.times_ms)

    @property
    def mean_ms(self) -> float:
        return self.total_ms / len(self.times_ms) if self.times_ms else 0.0

    @property
    def std_ms(self) -> float:
        if len(self.times_ms) < 2:
            return 0.0
        m = self.mean_ms
        return (sum((t - m) ** 2 for t in self.times_ms) / len(self.times_ms)) ** 0.5

    @property
    def n_batches(self) -> int:
        return len(self.times_ms)


def _sync(device: torch.device) -> None:
    """Synchronise accelerator before taking a timestamp."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def _load_module(
    vocab_path: str,
    vocab: LatexVocab,
    ckpt_path: Path,
    device: torch.device,
    *,
    use_kvcache: bool,
):
    """Instantiate a Lightning module and load checkpoint weights."""
    hparams: dict = {}
    ckpt = None
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        hparams = ckpt.get("hyper_parameters", {})

    in_channels = hparams.get("in_channels", 12)
    if ckpt is not None:
        state_dict = ckpt.get("state_dict", ckpt)
        for key in state_dict.keys():
            if key.endswith("conv1.weight"):
                in_channels = state_dict[key].shape[1]
                break

    model_cls = ExperimentalTransformerKVCache if use_kvcache else ExperimentalTransformer
    model = model_cls(
        in_channels=in_channels,
        vocab_size=len(vocab),
        pad_idx=vocab.PAD,
        d_model=hparams.get("d_model", 256),
        nhead=hparams.get("nhead", 8),
        num_encoder_layers=hparams.get("num_encoder_layers", 4),
        num_decoder_layers=hparams.get("num_decoder_layers", 4),
        dim_feedforward=hparams.get("dim_feedforward", 1024),
        dropout=hparams.get("dropout", 0.1),
    )

    module = HMELightningModule(vocab_path=vocab_path, model=model)

    if ckpt is not None:
        state_dict = ckpt.get("state_dict", ckpt)
        cleaned = {k.replace("._orig_mod.", "."): v for k, v in state_dict.items()}
        module.load_state_dict(cleaned, strict=False)

    return module.to(device).eval()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark inference: baseline vs KV-cache on real data."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device to run on (default: cpu).",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=10,
        help="Max number of test batches to benchmark (default: 10).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size for the test dataloader (default: {BATCH_SIZE}).",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=MAX_GENERATE_LEN,
        help=f"Max generation length (default: {MAX_GENERATE_LEN}).",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=str(CKPT_PATH),
        help=f"Path to Lightning checkpoint (default: {CKPT_PATH}).",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    vocab = LatexVocab.load(VOCAB_PATH)
    ckpt = Path(args.ckpt)

    factory = HMEDataLoaderFactory(
        root=ROOT,
        processed=False,
        vocab=vocab,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=False,
        min_len=None,
        max_len=None,
    )
    test_loader = factory.test()
    total_batches = len(test_loader)
    n_batches = min(args.num_batches, total_batches)

    print(f"\n  Loading checkpoint: {ckpt}")
    baseline = _load_module(str(VOCAB_PATH), vocab, ckpt, device, use_kvcache=False)
    kvcache = _load_module(str(VOCAB_PATH), vocab, ckpt, device, use_kvcache=True)

    baseline.max_generate_len = args.max_len
    kvcache.max_generate_len = args.max_len

    stats_base = RunStats("baseline")
    stats_kv = RunStats("kv-cache")

    print(f"\n{'═' * 60}")
    print(f"  Benchmark: Baseline vs KV-Cache  ({device})")
    print(f"  Batches: {n_batches}/{total_batches}  |  Batch size: {args.batch_size}")
    print(f"  Max generation length: {args.max_len}")
    print(f"{'═' * 60}\n")

    for batch_idx, batch in enumerate(test_loader):
        if batch_idx >= n_batches:
            break

        padded_features, feature_lengths, _padded_tokens, _token_lengths = batch
        src = padded_features.to(device)
        in_ch = baseline.model.conv1.weight.shape[1]
        if src.shape[2] > in_ch:
            src = src[:, :, :in_ch]
        src_lengths = feature_lengths.to(device)
        tag = f"batch {batch_idx + 1}/{n_batches}"

        _sync(device)
        t0 = time.perf_counter()
        baseline.generate(src, src_lengths)
        _sync(device)
        t1 = time.perf_counter()
        base_ms = (t1 - t0) * 1000
        stats_base.times_ms.append(base_ms)

        _sync(device)
        t0 = time.perf_counter()
        kvcache.generate(src, src_lengths)
        _sync(device)
        t1 = time.perf_counter()
        kv_ms = (t1 - t0) * 1000
        stats_kv.times_ms.append(kv_ms)

        speedup = base_ms / kv_ms if kv_ms > 0 else float("inf")
        print(
            f"  [{tag:>14}]  baseline: {base_ms:>8.1f} ms  |  "
            f"kv-cache: {kv_ms:>8.1f} ms  |  speedup: {speedup:.2f}x"
        )

    speedup_mean = stats_base.mean_ms / stats_kv.mean_ms if stats_kv.mean_ms > 0 else float("inf")
    speedup_total = (
        stats_base.total_ms / stats_kv.total_ms if stats_kv.total_ms > 0 else float("inf")
    )

    print(f"\n{'═' * 60}")
    print(f"  RESULTS  ({stats_base.n_batches} batches)")
    print(f"{'─' * 60}")
    print(f"  {'':>12} {'Mean (ms)':>12} {'Std (ms)':>10} {'Total (ms)':>12}")
    print(
        f"  {'baseline':>12} {stats_base.mean_ms:>12.1f} {stats_base.std_ms:>10.1f} {stats_base.total_ms:>12.1f}"
    )
    print(
        f"  {'kv-cache':>12} {stats_kv.mean_ms:>12.1f} {stats_kv.std_ms:>10.1f} {stats_kv.total_ms:>12.1f}"
    )
    print(f"{'─' * 60}")
    print(f"  Mean speedup:  {speedup_mean:.2f}x")
    print(f"  Total speedup: {speedup_total:.2f}x")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()
