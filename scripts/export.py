from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml
from lightning.pytorch.cli import instantiate_class
from onnxruntime.quantization import QuantType, quantize_dynamic

from hand_to_tex.models import HMELightningModule
from hand_to_tex.models.components.exportable import OnnxExportable
from hand_to_tex.utils import LatexVocab, logger
from hand_to_tex.utils.inference import onnx_batch_inference


def _load_lightning_module(
    ckpt_path: Path,
    config_path: Path,
    device: str = "cpu",
) -> tuple[HMELightningModule, LatexVocab]:
    """Create an `HMELightningModule` and load weights from a checkpoint.

    Parameters
    ----------
    ckpt_path:
        Path to the Lightning checkpoint.
    vocab_path:
        Path to the vocab used for training
    **hparams:
        Hyper parameters to overwrite, for example, `class-path` can be used for
        converting with modified architecture compatible with previous one

    Returns
    -------
    HMELightningModule
        Module in eval mode with checkpoint weights loaded.
    """
    with open(config_path, encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if (vocab_path := config["model"].get("vocab_path")) is None:
        vocab = LatexVocab.default()
    else:
        vocab = LatexVocab.load(path=vocab_path)

    model_config = config["model"]["model"]
    model_config["init_args"]["vocab_size"] = len(vocab)
    model_config["init_args"]["pad_idx"] = vocab.PAD

    model_instance = instantiate_class((), model_config)

    module = HMELightningModule.load_from_checkpoint(
        checkpoint_path=ckpt_path,
        weights_only=True,
        model=model_instance,
        strict=False,
        map_location=device,
    )

    module.eval()
    return module, vocab


def _quantize_onnx(src_path: Path, dst_path: Path) -> None:
    quantize_dynamic(
        model_input=str(src_path),
        model_output=str(dst_path),
        weight_type=QuantType.QInt8,
    )
    logger.info(f"Quantized ONNX model saved to {dst_path}")


def _get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a KV-cache capable checkpoint to ONNX and run test inference."
    )
    parser.add_argument(
        "--ckpt",
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
        "--out-dir",
        type=str,
        default="data/models/onnx/test",
        help="Directory, where the exported model-directory will be created",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Export INT8 quantized ONNX models and use them for inference",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Optional LightningCLI config.yaml used to resolve model class_path",
    )
    return parser


def main() -> None:
    parser = _get_parser()
    args = parser.parse_args()

    logger.info(f"Loading lightning module from ckpt={args.ckpt}, config={args.config}")
    module, vocab = _load_lightning_module(ckpt_path=Path(args.ckpt), config_path=Path(args.config))

    assert isinstance(module.model, OnnxExportable)

    out_dir = Path(args.out_dir)
    test_dir = Path(args.test_input)
    created_onnx_modules = module.export_to_onnx(out_dir=out_dir)

    if args.test_input:
        logger.info(f"Test input provided. Preparing to run inference on: {args.test_input}")

        onnx_batch_inference(
            model_class=type(module.model),
            inkml_directory=test_dir,
            model_directory=out_dir,
            vocab=vocab,
            max_len=150,
        )
        logger.info(f"Model saved to: {out_dir}")

    if args.quantize:
        quant_out_dir = out_dir.with_name(f"{out_dir.name}-quantized")
        os.makedirs(name=quant_out_dir, exist_ok=True)

        for module_name, src_path in created_onnx_modules.items():
            dst_path = quant_out_dir / f"{module_name}.onnx"
            _quantize_onnx(src_path=src_path, dst_path=dst_path)

        logger.info("Quantize flag provided, starting quantization and inference")

        onnx_batch_inference(
            model_class=type(module.model),
            inkml_directory=Path(args.test_input),
            model_directory=quant_out_dir,
            vocab=vocab,
            max_len=150,
        )
        logger.info(f"Model saved to: {quant_out_dir}")


if __name__ == "__main__":
    main()
