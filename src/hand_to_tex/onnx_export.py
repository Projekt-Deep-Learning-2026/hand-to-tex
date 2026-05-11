from __future__ import annotations

from pathlib import Path

import yaml
from lightning.pytorch.cli import instantiate_class
from onnxruntime.quantization import QuantType, quantize_dynamic

from hand_to_tex.models import HMELightningModule
from hand_to_tex.utils import LatexVocab, logger


def _load_lightning_module(
    ckpt_path: Path,
    config_path: Path,
    device: str = "cpu",
) -> HMELightningModule:
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
    return module


def _quantize_onnx(src_path: Path, dst_path: Path) -> None:
    quantize_dynamic(
        model_input=str(src_path),
        model_output=str(dst_path),
        weight_type=QuantType.QInt8,
    )
    logger.info(f"Quantized ONNX model saved to {dst_path}")


def export_onnx_from_ckpt(
    *,
    ckpt_path: Path,
    config_path: Path,
    quantize: bool,
) -> None:
    module = _load_lightning_module(ckpt_path=ckpt_path, config_path=config_path)
    print(module)
