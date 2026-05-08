from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import torch
import yaml
from torch import Tensor

from hand_to_tex.models.components.base import BaseDecoderModel
from hand_to_tex.models.lit_module import HMELightningModule
from hand_to_tex.utils import LatexVocab, logger


def collect_inkml_files(input_path: Path) -> list[Path]:
    """Collect `.inkml` files from a file or directory.

    Parameters
    ----------
    input_path:
        Either a single `.inkml` file or a directory containing `.inkml` files.

    Returns
    -------
    list[Path]
        Sorted list of matching files. A single file input is returned as a one-item list.

    Raises
    ------
    FileNotFoundError
        If `input_path` does not exist as a file or directory.
    """
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(input_path.rglob("*.inkml"))
    raise FileNotFoundError(f"Invalid input path: {input_path}")


def load_checkpoint(ckpt_path: Path) -> dict:
    """Load a Lightning checkpoint and return its hyper-parameters.

    Parameters
    ----------
    ckpt_path:
        Path to a `.ckpt` file produced by Lightning.

    Returns
    -------
    dict
        The checkpoint `hyper_parameters` mapping.
    """
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return ckpt.get("hyper_parameters", {})


def infer_in_channels(init_args: dict) -> int:
    """Read the input feature width from model init arguments.

    Parameters
    ----------
    init_args:
        Model init arguments loaded from checkpoint/config.
    Returns
    -------
    int
        The expected input feature width for the first convolution.

    Raises
    ------
    KeyError
        If `in_channels` is missing from the model metadata.
    """
    if "in_channels" not in init_args:
        raise KeyError("model init args must include 'in_channels'.")
    return int(init_args["in_channels"])


def load_config(config_path: Path | None) -> dict:
    """Load a YAML config file into a dictionary.

    Parameters
    ----------
    config_path:
        Path to a config file or `None`.

    Returns
    -------
    dict
        Parsed YAML content, or an empty dict when the file is absent.

    """
    if config_path is None or not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def extract_model_block(source: dict) -> dict | None:
    """Extract a Lightning-style model config block from a nested dictionary.

    The function looks for a top-level `class_path`, then for `model.class_path`,
    and finally for `model.model.class_path` because the project config stores the
    LightningModule and the decoder model in a nested structure.

    Parameters
    ----------
    source:
        Parsed checkpoint metadata or YAML configuration.

    Returns
    -------
    dict | None
        The nested mapping that contains `class_path` and optional `init_args`.
    """
    if not isinstance(source, dict):
        return None

    if isinstance(source.get("class_path"), str):
        return source

    model_cfg = source.get("model")
    if isinstance(model_cfg, dict):
        if isinstance(model_cfg.get("class_path"), str):
            return model_cfg
        inner = model_cfg.get("model")
        if isinstance(inner, dict) and isinstance(inner.get("class_path"), str):
            return inner

    return None


def resolve_model_spec(hparams: dict, config_path: Path | None) -> tuple[str, dict]:
    """Resolve the decoder class path and init args from checkpoint or config.

    Parameters
    ----------
    hparams:
        Hyper-parameters loaded from the checkpoint.
    config_path:
        Optional LightningCLI config file used as a fallback source.

    Returns
    -------
    tuple[str, dict]
        A pair `(class_path, init_args)` suitable for dynamic model loading.

    """
    model_block = extract_model_block(hparams)
    source_label = "checkpoint"

    if model_block is None and config_path is not None:
        config_data = load_config(config_path)
        model_block = extract_model_block(config_data)
        source_label = f"config ({config_path})"

    if model_block is None:
        raise RuntimeError(
            "No model.class_path found in checkpoint hyper_parameters or config file."
        )

    class_path = model_block["class_path"]
    init_args = model_block.get("init_args", {})

    logger.info(f"Using model class_path from {source_label}: {class_path}")
    return class_path, init_args


def import_model_class(class_path: str) -> type[BaseDecoderModel]:
    """Import a decoder class by fully qualified name and validate the base type.

    Parameters
    ----------
    class_path:
        A dotted path such as `package.module.ClassName`.

    Returns
    -------
    type[BaseDecoderModel]
        Imported class that is guaranteed to inherit from `BaseDecoderModel`.
    """
    if not class_path or "." not in class_path:
        raise ValueError("`class_path` must be a full module path like 'pkg.module.ClassName'.")

    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    model_cls = getattr(module, class_name, None)
    if not isinstance(model_cls, type):
        raise TypeError(f"Resolved object for '{class_path}' is not a class.")
    if not issubclass(model_cls, BaseDecoderModel):
        raise TypeError(f"'{class_path}' does not extend BaseDecoderModel.")
    return model_cls


def filter_init_args(model_cls: type[BaseDecoderModel], init_args: dict) -> dict:
    """Filter a config dictionary down to arguments accepted by a constructor.

    Parameters
    ----------
    model_cls:
        The model class whose `__init__` signature will be inspected.
    init_args:
        Dictionary of candidate keyword arguments.

    Returns
    -------
    dict
        A filtered dictionary containing only constructor-compatible keys.
    """
    sig = inspect.signature(model_cls.__init__)
    if any(param.kind == param.VAR_KEYWORD for param in sig.parameters.values()):
        return dict(init_args)
    allowed = {name for name in sig.parameters if name != "self"}
    return {key: value for key, value in init_args.items() if key in allowed}


def build_decoder_model(
    model_cls: type[BaseDecoderModel],
    vocab: LatexVocab,
    init_args: dict,
) -> BaseDecoderModel:
    """Instantiate a decoder model from checkpoint metadata.

    Parameters
    ----------
    model_cls:
        The decoder class to instantiate.
    vocab:
        Loaded vocabulary used to derive `vocab_size` and `pad_idx`.
    init_args:
        Constructor arguments extracted from checkpoint/config metadata.

    Returns
    -------
    BaseDecoderModel
        Instantiated decoder model ready to be loaded with weights.
    """
    in_channels = infer_in_channels(init_args)
    model_args = {
        **init_args,
        "in_channels": in_channels,
        "vocab_size": len(vocab),
        "pad_idx": vocab.PAD,
    }
    filtered_args = filter_init_args(model_cls, model_args)
    return model_cls(**filtered_args)


def load_lightning_module(
    ckpt_path: Path,
    vocab_path: Path,
    vocab: LatexVocab,
    model_cls: type[BaseDecoderModel],
    init_args: dict,
) -> HMELightningModule:
    """Create an `HMELightningModule` and load weights from a checkpoint.

    Parameters
    ----------
    ckpt_path:
        Path to the Lightning checkpoint.
    vocab_path:
        Path to the vocabulary file passed into the Lightning module.
    vocab:
        Loaded vocabulary used for model construction.
    model_cls:
        The decoder model class to instantiate.
    init_args:
        Constructor arguments extracted from metadata.

    Returns
    -------
    HMELightningModule
        Module in eval mode with checkpoint weights loaded.
    """
    model = build_decoder_model(model_cls, vocab, init_args)

    module = HMELightningModule.load_from_checkpoint(
        ckpt_path,
        vocab_path=str(vocab_path),
        model=model,
        pretrained_model_path=None,
        strict=False,
    )

    module.eval()
    return module


def fit_features(features: Tensor, in_channels: int) -> Tensor:
    """Clip or pad a feature tensor to match a model's input width.

    Parameters
    ----------
    features:
        Input tensor shaped `(T, F)`.
    in_channels:
        Target number of feature channels expected by the model.

    Returns
    -------
    Tensor
        Tensor shaped `(T, in_channels)`.

    Notes
    -----
    The operation is symmetric with respect to width mismatch: extra channels are
    dropped and missing channels are padded with zeros on the right.
    """
    if features.shape[1] > in_channels:
        return features[:, :in_channels]
    if features.shape[1] < in_channels:
        pad = torch.zeros(
            (features.shape[0], in_channels - features.shape[1]),
            dtype=features.dtype,
            device=features.device,
        )
        return torch.cat([features, pad], dim=1)
    return features


def resolve_config_path(config_path: Path | None) -> Path | None:
    """Resolve the config file used to recover model metadata.

    Parameters
    ----------
    config_path:
        Explicit config path supplied by the caller. If `None`, the repository
        root `config.yaml` is used when present.

    Returns
    -------
    Path | None
        Resolved config path, or `None` if no config file is available.

    """
    if config_path is None:
        default_config = Path("config.yaml")
        return default_config if default_config.exists() else None
    return config_path
