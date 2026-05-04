import argparse
import time
from pathlib import Path

import torch
from torch import Tensor

from hand_to_tex.datasets.dataset import _HMEDatasetBase
from hand_to_tex.datasets.ink_data import InkData
from hand_to_tex.models.lit_module import HMELightningModule
from hand_to_tex.utils import logger


def _collect_inputs(input_path: Path, max_samples: int | None) -> list[Tensor]:
    if input_path.is_file():
        inkml_files = [input_path]
    elif input_path.is_dir():
        inkml_files = sorted(input_path.rglob("*.inkml"))
    else:
        raise FileNotFoundError(f"Invalid input path: {input_path}")

    if max_samples is not None:
        inkml_files = inkml_files[:max_samples]

    if not inkml_files:
        raise FileNotFoundError(f"No .inkml files found in {input_path}")

    features_list = []
    for inkml_path in inkml_files:
        ink = InkData.load(inkml_path)
        features = _HMEDatasetBase.extract_features(ink)
        if features.size(0) == 0:
            logger.warning("Skipping empty trace file: %s", inkml_path.name)
            continue
        features_list.append(features)

    if not features_list:
        raise ValueError("All provided inkml files were empty.")

    return features_list


def _select_device(device_str: str | None) -> torch.device:
    if device_str:
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _run_benchmark(
    model: HMELightningModule,
    features_list: list[Tensor],
    device: torch.device,
    warmup: int,
    runs: int,
) -> float:
    model.eval()
    model.to(device)

    def _infer_once() -> None:
        for features in features_list:
            features_batched = features.unsqueeze(0).to(device)
            lengths = torch.tensor([features.size(0)], dtype=torch.long, device=device)
            _ = model.generate(src=features_batched, src_lengths=lengths)

    with torch.inference_mode():
        for _ in range(warmup):
            _infer_once()

        start = time.perf_counter()
        for _ in range(runs):
            _infer_once()
        end = time.perf_counter()

    return (end - start) / runs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark inference speed for baseline vs KV-cache"
    )
    parser.add_argument("--ckpt", type=str, required=True, help="checkpoint .ckpt file path")
    parser.add_argument("--input", type=str, required=True, help="Path to .inkml file or directory")
    parser.add_argument(
        "--vocab", type=str, default="data/assets/vocab.json", help="Vocabulary .json file path"
    )
    parser.add_argument("--device", type=str, default=None, help="Override device (cpu/cuda/mps)")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup iterations")
    parser.add_argument("--runs", type=int, default=3, help="Benchmark iterations")
    parser.add_argument("--max-samples", type=int, default=10, help="Max files to benchmark")
    args = parser.parse_args()

    input_path = Path(args.input)
    device = _select_device(args.device)
    features_list = _collect_inputs(input_path, args.max_samples)

    logger.info("Device: %s", device)
    logger.info("Samples: %d", len(features_list))

    logger.info("Loading baseline model...")
    baseline = HMELightningModule.load_from_checkpoint(
        checkpoint_path=args.ckpt,
        vocab_path=args.vocab,
        map_location=device,
        pretrained_model_path=None,
        weights_only=True,
        use_kvcache=False,
    )

    logger.info("Loading KV-cache model...")
    kvcache = HMELightningModule.load_from_checkpoint(
        checkpoint_path=args.ckpt,
        vocab_path=args.vocab,
        map_location=device,
        pretrained_model_path=None,
        weights_only=True,
        use_kvcache=True,
    )

    base_time = _run_benchmark(baseline, features_list, device, args.warmup, args.runs)
    kv_time = _run_benchmark(kvcache, features_list, device, args.warmup, args.runs)

    logger.info("Baseline avg: %.4fs", base_time)
    logger.info("KV-cache avg: %.4fs", kv_time)
    logger.info("Speedup: %.2fx", base_time / kv_time if kv_time > 0 else float("inf"))


if __name__ == "__main__":
    main()
