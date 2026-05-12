import os
from pathlib import Path

import onnxruntime as ort
import torch

from hand_to_tex.datasets import InkData
from hand_to_tex.datasets.dataset import _HMEDatasetBase
from hand_to_tex.models.components.exportable import OnnxExportable
from hand_to_tex.types import BatchedFeatures, FeatureLengths, Features
from hand_to_tex.utils import LatexVocab, logger


def gather_inkmls(directory: Path) -> list[Path]:
    """Collect `.inkml` files from a file or directory.

    Parameters
    ----------
    directory:
        Either a single `.inkml` file or a directory containing `.inkml` files.

    Returns
    -------
    list[Path]
        Sorted list of matching files. A single file input is returned as a one-item list.

    Raises
    ------
    FileNotFoundError
        If `directory` does not exist, is not a valid inkml file, or a directory.
    """

    if directory.is_file() and directory.suffix == "inkml":
        return [directory]
    elif directory.is_dir():
        return sorted(directory.rglob("*.inkml"))

    raise FileNotFoundError(f"Invalid directory for gathering inkml files: {directory}")


def onnx_batch_inference(
    model_class: type[OnnxExportable],
    inkml_directory: Path,
    model_directory: Path,
    vocab: LatexVocab,
    max_len: int,
) -> None:
    inkml_files = gather_inkmls(directory=inkml_directory)

    inks: list[InkData] = [InkData.load(path=pth) for pth in inkml_files]

    sessions = {
        Path(pth).stem: ort.InferenceSession(
            model_directory / pth, providers=["CPUExecutionProvider"]
        )
        for pth in os.listdir(model_directory)
        if pth.endswith(".onnx")
    }

    logger.info(f"Starting Inference with: {', '.join(sessions.keys())}")
    for ink in inks:
        fts: Features = _HMEDatasetBase.extract_features(ink=ink)

        if fts.numel() == 0:
            logger.info(f"Skipping empty sample - {ink.sample_id}")
            continue

        fts_batched: BatchedFeatures = fts.unsqueeze(0)
        lengths: FeatureLengths = torch.tensor([fts.size(0)], dtype=torch.long)

        predicted_tokens = model_class.run_onnx_inference(
            sessions=sessions,
            src_features=fts_batched,
            src_lengths=lengths,
            vocab=vocab,
            max_len=max_len,
        )

        expected_tokens = vocab.decode_sequence(vocab.encode_expr(ink.tex_norm))

        logger.info("=" * 72)
        logger.info(f"Sample:    {ink.sample_id}")
        logger.info(f"Expected:  {' '.join(expected_tokens[1:-1])}")
        logger.info(f"Predicted: {' '.join(predicted_tokens[0][1:-1])}")
