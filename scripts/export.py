from __future__ import annotations

import argparse
from pathlib import Path

from hand_to_tex.onnx_export import export_onnx_from_ckpt


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a KV-cache capable checkpoint to ONNX and run test inference."
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        required=True,
        help="Path to the .ckpt file with model's state dict",
    )
    parser.add_argument(
        "--test-input",
        type=str,
        required=False,
        help="Path to a .inkml file or directory of .inkml files",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Export INT8 quantized ONNX models and use them for inference",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        required=True,
        help="Optional LightningCLI config.yaml used to resolve model class_path",
    )

    args = parser.parse_args()

    export_onnx_from_ckpt(
        ckpt_path=Path(args.ckpt_path),
        quantize=args.quantize,
        config_path=Path(args.config_path),
    )


if __name__ == "__main__":
    main()
