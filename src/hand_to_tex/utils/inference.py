from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from hand_to_tex.datasets import InkData
from hand_to_tex.datasets.dataset import _HMEDatasetBase
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
    inkml_directory: Path,
    vocab_path: Path,
    encoder_path: Path,
    decoder_path: Path,
    max_len: int,
) -> None:
    inkml_files = gather_inkmls(directory=inkml_directory)
    vocab = LatexVocab.load(path=vocab_path)

    logger.info(
        f"Running onnx inference decoder={decoder_path} | encoder={encoder_path} | {len(inkml_files)} files"
    )

    inks: list[InkData] = [InkData.load(path=pth) for pth in inkml_files]

    enc_session = ort.InferenceSession(encoder_path, providers=["CPUExecutionProvider"])
    dec_session = ort.InferenceSession(decoder_path, providers=["CPUExecutionProvider"])

    logger.info("Starting Inference")
    for ink in inks:
        fts: Features = _HMEDatasetBase.extract_features(ink=ink)

        if fts.numel() == 0:
            logger.info(f"Skipping empty sample - {ink.sample_id}")
            continue

        fts_batched: BatchedFeatures = fts.unsqueeze(0)
        lengths: FeatureLengths = torch.tensor([fts.size(0)], dtype=torch.long)

        _memory, mem_mask, mem_k, mem_v = enc_session.run(
            ["memory", "mem_mask", "mem_k", "mem_v"],
            {
                "src": fts_batched.detach().cpu().numpy().astype(np.float32),
                "src_lengths": lengths.detach().cpu().numpy().astype(np.int64),
            },
        )

        mem_mask = np.asarray(mem_mask)
        mem_k = np.asarray(mem_k)
        mem_v = np.asarray(mem_v)

        generated_tokens = _onnx_generate_cached(
            session=dec_session,
            mem_k=mem_k,
            mem_v=mem_v,
            mem_mask=mem_mask,
            sos_idx=vocab.SOS,
            eos_idx=vocab.EOS,
            pad_idx=vocab.PAD,
            max_len=max_len,
        )

        predicted_tokens = vocab.decode_sequence(token_ids=generated_tokens[0].tolist())
        expected_tokens = vocab.decode_sequence(vocab.encode_expr(ink.tex_norm))

        logger.info("=" * 72)
        logger.info(f"Sample:    {ink.sample_id}")
        logger.info(f"Expected:  {' '.join(expected_tokens[1:-1])}")
        logger.info(f"Predicted: {' '.join(predicted_tokens[1:-1])}")


def _onnx_generate_cached(
    session: ort.InferenceSession,
    mem_k: np.ndarray,
    mem_v: np.ndarray,
    mem_mask: np.ndarray,
    *,
    sos_idx: int,
    eos_idx: int,
    pad_idx: int,
    max_len: int,
) -> np.ndarray:

    for inp in session.get_inputs():
        if inp.name == "self_k":
            shape = inp.shape
            num_layers = shape[0]
            num_heads = shape[2]
            head_dim = shape[4]
            break
    else:
        raise RuntimeError("No `self_k` in encoder graph")

    mem_k_np = mem_k.astype(np.float32, copy=False)
    mem_v_np = mem_v.astype(np.float32, copy=False)
    mem_mask_np = mem_mask.astype(np.bool_, copy=False)

    batch_size = mem_k_np.shape[1]
    tgt = np.full((batch_size, 1), sos_idx, dtype=np.int64)
    unfinished = np.ones(batch_size, dtype=bool)
    step = np.array([0], dtype=np.int64)

    self_k = np.zeros((num_layers, batch_size, num_heads, 1, head_dim), dtype=np.float32)
    self_v = np.zeros((num_layers, batch_size, num_heads, 1, head_dim), dtype=np.float32)

    for _ in range(1, max_len):
        logits, self_k, self_v = session.run(
            ["logits", "self_k_out", "self_v_out"],
            {
                "tgt_last": tgt[:, -1:],
                "mem_k": mem_k_np,
                "mem_v": mem_v_np,
                "mem_mask": mem_mask_np,
                "step": step,
                "self_k": self_k,
                "self_v": self_v,
            },
        )

        next_token = np.argmax(logits, axis=-1).astype(np.int64)
        next_step = np.where(unfinished, next_token, pad_idx)
        tgt = np.concatenate([tgt, next_step[:, None]], axis=1)
        unfinished = np.logical_and(unfinished, next_token != eos_idx)

        if not unfinished.any():
            return tgt

        step = step + 1

    return tgt
