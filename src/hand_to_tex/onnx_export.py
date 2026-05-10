from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from onnxruntime.quantization import QuantType, quantize_dynamic
from torch import Tensor

from hand_to_tex.models.components.base import BaseDecoderModel
from hand_to_tex.models.lit_module import HMELightningModule
from hand_to_tex.utils import LatexVocab, logger
from hand_to_tex.utils.export_helpers import (
    import_model_class,
    load_checkpoint,
    load_lightning_module,
    resolve_config_path,
    resolve_model_spec,
)
from hand_to_tex.utils.inference import onnx_batch_inference


def _require_kvcache_model(model: BaseDecoderModel) -> Any:
    # Keep checks minimal: ensure the model has a transformer and d_model.
    if not hasattr(model, "transformer") or not hasattr(model, "d_model"):
        raise TypeError(
            "Export requires a KV-cache-capable model with 'transformer' and 'd_model'."
        )

    tgt_pe = getattr(model, "tgt_pe", None)
    if tgt_pe is None or not hasattr(tgt_pe, "pe") or not hasattr(tgt_pe, "dropout"):
        raise TypeError("Export requires model.tgt_pe with 'pe' and 'dropout'.")

    return model


class HTTEncoderOnnx(nn.Module):
    """ONNX-friendly encoder that avoids Transformer MHA reshape pitfalls."""

    def __init__(self, model: BaseDecoderModel) -> None:
        super().__init__()
        self.model = _require_kvcache_model(model)

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


# The full non-cached decoder implementation was unused in export flow and
# removed to keep this module focused on encoder + cached decoder-step.


class HTTDecoderStepOnnx(nn.Module):
    """ONNX-friendly cached decoder step with self-attention KV reuse."""

    def __init__(self, model: BaseDecoderModel) -> None:
        super().__init__()
        self.model = _require_kvcache_model(model)

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
    vocab_path: Path,
    test_input: Path,
    max_len: int | None,
    quantize: bool,
    config_path: Path | None = None,
) -> None:
    vocab = LatexVocab.load(vocab_path)

    hparams = load_checkpoint(ckpt_path)
    print(hparams)
    config_path = resolve_config_path(config_path)
    class_path, init_args = resolve_model_spec(hparams, config_path)
    model_cls = import_model_class(class_path)
    module = load_lightning_module(
        ckpt_path=ckpt_path,
        vocab_path=vocab_path,
        vocab=vocab,
        model_cls=model_cls,
        init_args=init_args,
    )
    in_channels = int(init_args["in_channels"])

    decoder_layers = module.model.transformer.decoder.layers
    if not decoder_layers:
        raise RuntimeError("Decoder has no layers; cannot build cached ONNX decoder.")
    num_heads = decoder_layers[0].self_attn.num_heads
    if module.model.d_model % num_heads != 0:
        raise ValueError("d_model must be divisible by num_heads for cached decoder export")

    resolved_max_len = module.max_generate_len if max_len is None else max_len
    if resolved_max_len < 2:
        raise ValueError("--max-len must be >= 2")

    encoder_path = ckpt_path.with_name(f"{ckpt_path.stem}_encoder.onnx")
    decoder_path = ckpt_path.with_name(f"{ckpt_path.stem}_decoder_step.onnx")
    _export_onnx(module, encoder_path, decoder_path, in_channels)

    encoder_eval_path = encoder_path
    decoder_eval_path = decoder_path
    if quantize:
        encoder_q_path = ckpt_path.with_name(f"{ckpt_path.stem}_encoder_int8.onnx")
        decoder_q_path = ckpt_path.with_name(f"{ckpt_path.stem}_decoder_step_int8.onnx")
        _quantize_onnx(encoder_path, encoder_q_path)
        _quantize_onnx(decoder_path, decoder_q_path)
        encoder_eval_path = encoder_q_path
        decoder_eval_path = decoder_q_path

    onnx_batch_inference(
        test_input,
        vocab_path,
        encoder_eval_path,
        decoder_eval_path,
        resolved_max_len,
    )
