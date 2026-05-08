from __future__ import annotations

import argparse
from pathlib import Path

from hand_to_tex.onnx_export import export_onnx_from_ckpt


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a KV-cache capable checkpoint to ONNX and run test inference."
    )
    parser.add_argument("--ckpt-path", type=str, required=True, help="Path to the .ckpt file")
    parser.add_argument(
        "--vocab-path",
        type=str,
        default="data/assets/vocab.json",
        help="Path to vocab.json",
    )
    parser.add_argument(
        "--test-input",
        type=str,
        required=True,
        help="Path to a .inkml file or directory of .inkml files",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=None,
        help="Max generation length override (defaults to checkpoint value)",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Export INT8 quantized ONNX models and use them for inference",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Optional LightningCLI config.yaml used to resolve model class_path",
    )

    args = parser.parse_args()

    export_onnx_from_ckpt(
        ckpt_path=Path(args.ckpt_path),
        vocab_path=Path(args.vocab_path),
        test_input=Path(args.test_input),
        max_len=args.max_len,
        quantize=args.quantize,
        config_path=Path(args.config_path) if args.config_path else None,
    )


if __name__ == "__main__":
    main()
