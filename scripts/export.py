from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from onnxruntime.quantization import QuantType, quantize_dynamic
from torch import Tensor

from hand_to_tex.datasets.dataset import _HMEDatasetBase
from hand_to_tex.datasets.ink_data import InkData
from hand_to_tex.models.components import ExperimentalTransformerKVCache
from hand_to_tex.models.lit_module import HMELightningModule
from hand_to_tex.utils import LatexVocab, logger


class HTTEncoderOnnx(nn.Module):
    """ONNX-friendly encoder that avoids Transformer MHA reshape pitfalls."""

    def __init__(self, model: ExperimentalTransformerKVCache) -> None:
        super().__init__()
        self.model = model

    def _build_mem_mask(self, src_features: Tensor, src_lengths: Tensor) -> Tensor:
        conv1_len = self.model._conv1d_output_lengths(src_lengths, self.model.conv1)
        conv2_len = self.model._conv1d_output_lengths(conv1_len, self.model.conv2)
        downsampled_lengths = conv2_len.clamp_min(0)

        ones = torch.ones_like(src_features[:, :, 0], dtype=torch.long)
        steps = torch.cumsum(ones, dim=1) - 1
        return steps >= downsampled_lengths.unsqueeze(1)

    def forward(self, src: Tensor, src_lengths: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        src_conv = src.transpose(1, 2)
        src_features = self.model.input_proj(src_conv).transpose(1, 2)
        src_emb = self.model.src_pe(src_features)

        mem_mask = self._build_mem_mask(src_features, src_lengths)

        x = src_emb
        for layer in self.model.transformer.encoder.layers:
            if not layer.norm_first:
                raise RuntimeError("ONNX encoder expects norm_first=True.")

            sa_input = layer.norm1(x)
            q_raw = self.model._project_q(layer.self_attn, sa_input)
            k_raw, v_raw = self.model._project_kv(layer.self_attn, sa_input)

            num_heads = layer.self_attn.num_heads
            q = self.model._split_heads(q_raw, num_heads)
            k = self.model._split_heads(k_raw, num_heads)
            v = self.model._split_heads(v_raw, num_heads)

            sa_out = self.model._scaled_dot_product_attention(
                q=q,
                k=k,
                v=v,
                out_proj=layer.self_attn.out_proj,
                key_padding_mask=mem_mask,
            )
            x = x + layer.dropout1(sa_out)

            ff_input = layer.norm2(x)
            ff_out = layer.linear2(layer.dropout(layer.activation(layer.linear1(ff_input))))
            x = x + layer.dropout2(ff_out)

        if self.model.transformer.encoder.norm is not None:
            x = self.model.transformer.encoder.norm(x)

        memory = x
        mem_k = []
        mem_v = []
        for layer in self.model.transformer.decoder.layers:
            num_heads = layer.multihead_attn.num_heads
            mem_k_raw, mem_v_raw = self.model._project_kv(layer.multihead_attn, memory)
            mem_k.append(self.model._split_heads(mem_k_raw, num_heads))
            mem_v.append(self.model._split_heads(mem_v_raw, num_heads))

        return memory, mem_mask, torch.stack(mem_k, dim=0), torch.stack(mem_v, dim=0)


class HTTDecoderOnnx(nn.Module):
    """ONNX-friendly decoder using custom attention and dynamic lengths."""

    def __init__(self, model: ExperimentalTransformerKVCache) -> None:
        super().__init__()
        self.model = model

    @staticmethod
    def _build_causal_mask(tgt: Tensor) -> Tensor:
        ones = torch.ones_like(tgt, dtype=torch.long)
        steps = torch.cumsum(ones, dim=1) - 1
        query_pos = steps[:, :, None]
        key_pos = steps[:, None, :]
        return key_pos > query_pos

    def _scaled_dot_product_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        out_proj: nn.Linear,
        key_padding_mask: Tensor | None = None,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        head_dim = q.size(-1)
        scale = head_dim**-0.5
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(
                attn_mask[:, None, :, :],
                torch.finfo(attn_scores.dtype).min,
            )

        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(
                key_padding_mask[:, None, None, :],
                torch.finfo(attn_scores.dtype).min,
            )

        attn_weights = torch.softmax(attn_scores, dim=-1)
        context = torch.matmul(attn_weights, v)
        merged = self.model._merge_heads(context)
        return out_proj(merged)

    def forward(self, tgt: Tensor, memory: Tensor, mem_mask: Tensor) -> Tensor:
        x = self.model.tgt_tok_emb(tgt) * math.sqrt(self.model.d_model)
        x = self.model.tgt_pe(x)

        causal_mask = self._build_causal_mask(tgt)

        for layer in self.model.transformer.decoder.layers:
            if not layer.norm_first:
                raise RuntimeError("ONNX decoder expects norm_first=True.")

            sa_input = layer.norm1(x)
            q_raw = self.model._project_q(layer.self_attn, sa_input)
            k_raw, v_raw = self.model._project_kv(layer.self_attn, sa_input)

            num_heads = layer.self_attn.num_heads
            q = self.model._split_heads(q_raw, num_heads)
            k = self.model._split_heads(k_raw, num_heads)
            v = self.model._split_heads(v_raw, num_heads)

            sa_out = self._scaled_dot_product_attention(
                q=q,
                k=k,
                v=v,
                out_proj=layer.self_attn.out_proj,
                attn_mask=causal_mask,
            )
            x = x + layer.dropout1(sa_out)

            ca_input = layer.norm2(x)
            q_cross_raw = self.model._project_q(layer.multihead_attn, ca_input)
            q_cross = self.model._split_heads(q_cross_raw, num_heads)

            mem_k_raw, mem_v_raw = self.model._project_kv(layer.multihead_attn, memory)
            mem_k = self.model._split_heads(mem_k_raw, num_heads)
            mem_v = self.model._split_heads(mem_v_raw, num_heads)

            ca_out = self._scaled_dot_product_attention(
                q=q_cross,
                k=mem_k,
                v=mem_v,
                out_proj=layer.multihead_attn.out_proj,
                key_padding_mask=mem_mask,
            )
            x = x + layer.dropout2(ca_out)

            ff_input = layer.norm3(x)
            ff_out = layer.linear2(layer.dropout(layer.activation(layer.linear1(ff_input))))
            x = x + layer.dropout3(ff_out)

        if self.model.transformer.decoder.norm is not None:
            x = self.model.transformer.decoder.norm(x)

        return self.model.fc_out(x)


class HTTDecoderStepOnnx(nn.Module):
    """ONNX-friendly cached decoder step with self-attention KV reuse."""

    def __init__(self, model: ExperimentalTransformerKVCache) -> None:
        super().__init__()
        self.model = model

    def _positional_step(self, step: Tensor) -> Tensor:
        step = step.to(dtype=torch.long).reshape(1)
        pe = self.model.tgt_pe.pe.squeeze(0)
        pe_step = torch.index_select(pe, dim=0, index=step).unsqueeze(0)
        return pe_step

    def forward(
        self,
        tgt_last: Tensor,
        mem_k: Tensor,
        mem_v: Tensor,
        mem_mask: Tensor,
        step: Tensor,
        self_k: Tensor,
        self_v: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        x = self.model.tgt_tok_emb(tgt_last) * math.sqrt(self.model.d_model)
        x = x + self._positional_step(step)
        x = self.model.tgt_pe.dropout(x)

        new_self_k = []
        new_self_v = []

        for layer_idx, layer in enumerate(self.model.transformer.decoder.layers):
            if not layer.norm_first:
                raise RuntimeError("ONNX decoder-step expects norm_first=True.")

            k_prev = self_k[layer_idx]
            v_prev = self_v[layer_idx]

            sa_input = layer.norm1(x)
            q_raw = self.model._project_q(layer.self_attn, sa_input)
            k_new_raw, v_new_raw = self.model._project_kv(layer.self_attn, sa_input)

            num_heads = layer.self_attn.num_heads
            q = self.model._split_heads(q_raw, num_heads)
            k_new = self.model._split_heads(k_new_raw, num_heads)
            v_new = self.model._split_heads(v_new_raw, num_heads)

            k_cat = torch.cat([k_prev, k_new], dim=2)
            v_cat = torch.cat([v_prev, v_new], dim=2)

            self_mask_base = torch.ones_like(k_cat[:, 0, :, 0], dtype=torch.long)
            self_steps = torch.cumsum(self_mask_base, dim=1) - 1
            self_mask = self_steps == 0

            sa_out = self.model._scaled_dot_product_attention(
                q=q,
                k=k_cat,
                v=v_cat,
                out_proj=layer.self_attn.out_proj,
                key_padding_mask=self_mask,
            )
            x = x + layer.dropout1(sa_out)

            ca_input = layer.norm2(x)
            q_cross_raw = self.model._project_q(layer.multihead_attn, ca_input)
            q_cross = self.model._split_heads(q_cross_raw, num_heads)

            ca_out = self.model._scaled_dot_product_attention(
                q=q_cross,
                k=mem_k[layer_idx],
                v=mem_v[layer_idx],
                out_proj=layer.multihead_attn.out_proj,
                key_padding_mask=mem_mask,
            )
            x = x + layer.dropout2(ca_out)

            ff_input = layer.norm3(x)
            ff_out = layer.linear2(layer.dropout(layer.activation(layer.linear1(ff_input))))
            x = x + layer.dropout3(ff_out)

            new_self_k.append(k_cat)
            new_self_v.append(v_cat)

        if self.model.transformer.decoder.norm is not None:
            x = self.model.transformer.decoder.norm(x)

        logits = self.model.fc_out(x)[:, -1, :]
        return logits, torch.stack(new_self_k, dim=0), torch.stack(new_self_v, dim=0)


def _collect_inkml_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(input_path.rglob("*.inkml"))
    raise FileNotFoundError(f"Invalid input path: {input_path}")


def _load_checkpoint(ckpt_path: Path) -> tuple[dict, dict]:
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hparams = ckpt.get("hyper_parameters", {})
    state_dict = ckpt.get("state_dict", ckpt)
    return hparams, state_dict


def _infer_in_channels(hparams: dict, state_dict: dict) -> int:
    in_channels = hparams.get("in_channels", _HMEDatasetBase.FEATURES)
    for key, value in state_dict.items():
        if key.endswith("conv1.weight"):
            in_channels = value.shape[1]
            break
    return in_channels


def _build_kvcache_model(vocab: LatexVocab, hparams: dict, state_dict: dict):
    in_channels = _infer_in_channels(hparams, state_dict)
    return ExperimentalTransformerKVCache(
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


def _load_module(
    ckpt_path: Path,
    vocab_path: Path,
    vocab: LatexVocab,
    hparams: dict,
    state_dict: dict,
) -> HMELightningModule:
    model = _build_kvcache_model(vocab, hparams, state_dict)

    module = HMELightningModule.load_from_checkpoint(
        ckpt_path,
        vocab_path=str(vocab_path),
        model=model,
        pretrained_model_path=None,
        strict=False,
    )

    module.eval()
    return module


def _export_onnx(
    module: HMELightningModule,
    encoder_path: Path,
    decoder_path: Path,
    in_channels: int,
) -> None:
    encoder = HTTEncoderOnnx(module.model).eval()
    decoder = HTTDecoderStepOnnx(module.model).eval()

    device = torch.device("cpu")
    encoder.to(device)
    decoder.to(device)

    dummy_src_len = 64
    dummy_tgt_len = 1
    dummy_cache_len = 1

    src = torch.zeros((1, dummy_src_len, in_channels), dtype=torch.float32, device=device)
    src_lengths = torch.tensor([dummy_src_len], dtype=torch.long, device=device)
    tgt_last = torch.full((1, dummy_tgt_len), module.vocab.SOS, dtype=torch.long, device=device)
    step = torch.tensor([1], dtype=torch.long, device=device)

    num_layers = len(module.model.transformer.decoder.layers)
    num_heads = module.model.transformer.decoder.layers[0].self_attn.num_heads
    head_dim = module.model.d_model // num_heads

    self_k = torch.zeros(
        (num_layers, 1, num_heads, dummy_cache_len, head_dim),
        dtype=torch.float32,
        device=device,
    )
    self_v = torch.zeros(
        (num_layers, 1, num_heads, dummy_cache_len, head_dim),
        dtype=torch.float32,
        device=device,
    )

    _memory, mem_mask, mem_k, mem_v = encoder(src, src_lengths)

    torch.onnx.export(
        encoder,
        (src, src_lengths),
        str(encoder_path),
        input_names=["src", "src_lengths"],
        output_names=["memory", "mem_mask", "mem_k", "mem_v"],
        dynamic_axes={
            "src": {0: "batch", 1: "src_len"},
            "src_lengths": {0: "batch"},
            "memory": {0: "batch", 1: "src_len_down"},
            "mem_mask": {0: "batch", 1: "src_len_down"},
            "mem_k": {1: "batch", 3: "src_len_down"},
            "mem_v": {1: "batch", 3: "src_len_down"},
        },
        opset_version=17,
        do_constant_folding=True,
    )

    torch.onnx.export(
        decoder,
        (tgt_last, mem_k, mem_v, mem_mask, step, self_k, self_v),
        str(decoder_path),
        input_names=[
            "tgt_last",
            "mem_k",
            "mem_v",
            "mem_mask",
            "step",
            "self_k",
            "self_v",
        ],
        output_names=["logits", "self_k_out", "self_v_out"],
        dynamic_axes={
            "tgt_last": {0: "batch"},
            "mem_k": {1: "batch", 3: "src_len_down"},
            "mem_v": {1: "batch", 3: "src_len_down"},
            "mem_mask": {0: "batch", 1: "src_len_down"},
            "self_k": {1: "batch", 3: "cache_len"},
            "self_v": {1: "batch", 3: "cache_len"},
            "logits": {0: "batch"},
            "self_k_out": {1: "batch", 3: "cache_len_out"},
            "self_v_out": {1: "batch", 3: "cache_len_out"},
        },
        opset_version=17,
        do_constant_folding=True,
    )

    logger.info(f"Exported ONNX encoder to {encoder_path}")
    logger.info(f"Exported ONNX decoder-step to {decoder_path}")


def _fit_features(features: Tensor, in_channels: int) -> Tensor:
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


def _quantize_onnx(src_path: Path, dst_path: Path) -> None:
    quantize_dynamic(
        model_input=str(src_path),
        model_output=str(dst_path),
        weight_type=QuantType.QInt8,
    )
    logger.info(f"Quantized ONNX model saved to {dst_path}")


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
    num_layers: int,
    num_heads: int,
    head_dim: int,
) -> np.ndarray:
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


def _run_inkml_eval(
    inkml_files: list[Path],
    vocab: LatexVocab,
    in_channels: int,
    encoder_path: Path,
    decoder_path: Path,
    max_len: int,
    num_layers: int,
    num_heads: int,
    head_dim: int,
) -> None:
    logger.info(f"Running inference on {len(inkml_files)} .inkml files")

    encoder_session = ort.InferenceSession(str(encoder_path), providers=["CPUExecutionProvider"])
    decoder_session = ort.InferenceSession(str(decoder_path), providers=["CPUExecutionProvider"])

    for inkml_path in inkml_files:
        ink = InkData.load(inkml_path)
        features = _HMEDatasetBase.extract_features(ink)

        if features.numel() == 0:
            logger.warning(f"Skipping empty traces: {inkml_path.name}")
            continue

        features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        features = _fit_features(features, in_channels)
        features_batched = features.unsqueeze(0)
        lengths = torch.tensor([features.size(0)], dtype=torch.long)

        _memory, mem_mask, mem_k, mem_v = encoder_session.run(
            ["memory", "mem_mask", "mem_k", "mem_v"],
            {
                "src": features_batched.detach().cpu().numpy().astype(np.float32),
                "src_lengths": lengths.detach().cpu().numpy().astype(np.int64),
            },
        )

        generated_tokens = _onnx_generate_cached(
            session=decoder_session,
            mem_k=mem_k,
            mem_v=mem_v,
            mem_mask=mem_mask,
            sos_idx=vocab.SOS,
            eos_idx=vocab.EOS,
            pad_idx=vocab.PAD,
            max_len=max_len,
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
        )

        predicted_tokens = vocab.decode_sequence(token_ids=generated_tokens[0].tolist())
        expected_tokens = vocab.decode_sequence(vocab.encode_expr(ink.tex_norm))

        logger.info("=" * 72)
        logger.info(f"Sample:    {ink.sample_id}")
        logger.info(f"Expected:  {' '.join(expected_tokens[1:-1])}")
        logger.info(f"Predicted: {' '.join(predicted_tokens[1:-1])}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export an ExperimentalTransformerKVCache checkpoint to ONNX and run test inference."
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

    args = parser.parse_args()

    ckpt_path = Path(args.ckpt_path)
    vocab_path = Path(args.vocab_path)
    test_input = Path(args.test_input)

    vocab = LatexVocab.load(vocab_path)

    inkml_files = _collect_inkml_files(test_input)
    if not inkml_files:
        logger.warning(f"No .inkml files found in {test_input}")
        return

    hparams, state_dict = _load_checkpoint(ckpt_path)
    in_channels = _infer_in_channels(hparams, state_dict)

    module = _load_module(ckpt_path, vocab_path, vocab, hparams, state_dict)

    decoder_layers = module.model.transformer.decoder.layers
    if not decoder_layers:
        raise RuntimeError("Decoder has no layers; cannot build cached ONNX decoder.")
    num_layers = len(decoder_layers)
    num_heads = decoder_layers[0].self_attn.num_heads
    if module.model.d_model % num_heads != 0:
        raise ValueError("d_model must be divisible by num_heads for cached decoder export")
    head_dim = module.model.d_model // num_heads

    max_len = module.max_generate_len if args.max_len is None else args.max_len
    if max_len < 2:
        raise ValueError("--max-len must be >= 2")

    encoder_path = ckpt_path.with_name(f"{ckpt_path.stem}_encoder.onnx")
    decoder_path = ckpt_path.with_name(f"{ckpt_path.stem}_decoder_step.onnx")
    _export_onnx(module, encoder_path, decoder_path, in_channels)

    encoder_eval_path = encoder_path
    decoder_eval_path = decoder_path
    if args.quantize:
        encoder_q_path = ckpt_path.with_name(f"{ckpt_path.stem}_encoder_int8.onnx")
        decoder_q_path = ckpt_path.with_name(f"{ckpt_path.stem}_decoder_step_int8.onnx")
        _quantize_onnx(encoder_path, encoder_q_path)
        _quantize_onnx(decoder_path, decoder_q_path)
        encoder_eval_path = encoder_q_path
        decoder_eval_path = decoder_q_path

    st = time.time()

    _run_inkml_eval(
        inkml_files,
        vocab,
        in_channels,
        encoder_eval_path,
        decoder_eval_path,
        max_len,
        num_layers,
        num_heads,
        head_dim,
    )

    en = time.time()

    logger.info(f"Ran {len(inkml_files)} samples in {en - st}sec")


if __name__ == "__main__":
    main()

# TODO
# 1. Refactor komponentow w models, popraw inheritance itd.
# 2. zapisuj onnx do foldera z plikami encoder.onnx, decodex.onnx
# 3. popraw demo - ma też służyć jako benchmark, który można będzie wtedy wywalić, ma obsługiwać onnx
# 4. popraw export, ma nie zależeć od wybranego modelu, class_path ma się ładować z ckpt jesli się da, ma zależeć tylko od implementacji BaseDecoderModel
# 5. zacznij pracować nad stroną w JS
