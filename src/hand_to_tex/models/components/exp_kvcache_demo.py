import math

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from hand_to_tex.models.components.experimental import ExperimentalTransformer
from hand_to_tex.models.components.exportable import OnnxExportable, OnnxExportConfiguration
from hand_to_tex.types import (
    BatchedFeatures,
    BatchedTokens,
    DecoderKVCache,
    FeatureLengths,
    LayerKVCache,
    TensorBool,
)
from hand_to_tex.utils import LatexVocab


class ExperimentalTransformerKVCacheDemo(ExperimentalTransformer, OnnxExportable):
    """Transformer variant with KV-cache decoding."""

    def __init__(
        self,
        in_channels: int,
        vocab_size: int,
        pad_idx: int,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__(
            in_channels=in_channels,
            vocab_size=vocab_size,
            pad_idx=pad_idx,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self._d_model_scale = math.sqrt(self.d_model)
        self._decoder_layers = tuple(self.transformer.decoder.layers)
        self._decoder_num_heads = tuple(layer.self_attn.num_heads for layer in self._decoder_layers)
        self._decoder_head_dim = tuple(self.d_model // heads for heads in self._decoder_num_heads)
        self._decoder_norm = self.transformer.decoder.norm

    def _split_heads(self, x: Tensor, num_heads: int) -> Tensor:
        """Reshape `(B, T, D)` to `(B, H, T, Dh)` for multi-head attention."""
        batch_size, seq_len, model_dim = x.shape
        head_dim = model_dim // num_heads
        return x.view(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)

    def _merge_heads(self, x: Tensor) -> Tensor:
        """Reshape `(B, H, T, Dh)` back to `(B, T, D)`."""
        batch_size, num_heads, seq_len, head_dim = x.shape
        return x.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, num_heads * head_dim)

    @staticmethod
    def _project_q(attn: nn.MultiheadAttention, x: Tensor) -> Tensor:
        """Project query states using attention layer query weights."""
        w_q, _, _ = attn.in_proj_weight.chunk(3, dim=0)
        if attn.in_proj_bias is None:
            b_q = None
        else:
            b_q, _, _ = attn.in_proj_bias.chunk(3, dim=0)
        return nn.functional.linear(x, w_q, b_q)

    @staticmethod
    def _project_kv(attn: nn.MultiheadAttention, x: Tensor) -> tuple[Tensor, Tensor]:
        """Project key/value states using attention layer key/value weights."""
        _, w_k, w_v = attn.in_proj_weight.chunk(3, dim=0)
        if attn.in_proj_bias is None:
            b_k = None
            b_v = None
        else:
            _, b_k, b_v = attn.in_proj_bias.chunk(3, dim=0)
        k = nn.functional.linear(x, w_k, b_k)
        v = nn.functional.linear(x, w_v, b_v)
        return k, v

    def _scaled_dot_product_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        out_proj: nn.Linear,
        key_padding_mask: TensorBool | None = None,
    ) -> Tensor:
        """Compute attention for pre-projected `(B, H, T, Dh)` query/key/value tensors."""
        head_dim = q.size(-1)
        scale = head_dim**-0.5
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(
                key_padding_mask[:, None, None, :],
                torch.finfo(attn_scores.dtype).min,
            )

        attn_weights = torch.softmax(attn_scores, dim=-1)
        context = torch.matmul(attn_weights, v)
        merged = self._merge_heads(context)
        return out_proj(merged)

    @torch.no_grad()
    def init_kv_cache(self, memory: Tensor) -> DecoderKVCache:
        """Initialize decoder cache for incremental generation."""
        batch_size, _src_len, _ = memory.shape
        decoder_layers = self._decoder_layers
        layer_caches: list[LayerKVCache] = []

        for layer, num_heads, head_dim in zip(
            decoder_layers,
            self._decoder_num_heads,
            self._decoder_head_dim,
            strict=True,
        ):
            empty_self_k = torch.empty(
                (batch_size, num_heads, 0, head_dim),
                device=memory.device,
                dtype=memory.dtype,
            )
            empty_self_v = torch.empty(
                (batch_size, num_heads, 0, head_dim),
                device=memory.device,
                dtype=memory.dtype,
            )

            mem_k_raw, mem_v_raw = self._project_kv(layer.multihead_attn, memory)
            mem_k = self._split_heads(mem_k_raw, num_heads)
            mem_v = self._split_heads(mem_v_raw, num_heads)

            layer_caches.append(
                {
                    "self_k": empty_self_k,
                    "self_v": empty_self_v,
                    "mem_k": mem_k,
                    "mem_v": mem_v,
                }
            )

        return {"step": 0, "layers": layer_caches}

    @torch.no_grad()
    def decode_step(
        self,
        tgt_last: BatchedTokens,
        memory: Tensor,
        memory_key_padding_mask: TensorBool,
        cache: DecoderKVCache,
    ) -> tuple[Tensor, DecoderKVCache]:
        """Decode a single autoregressive step using cached projected K/V tensors."""

        step = cache["step"]
        layer_caches = cache["layers"]

        x = self.tgt_tok_emb(tgt_last) * self._d_model_scale
        x = self.tgt_pe.forward_step(x, step)

        decoder_layers = self._decoder_layers

        for layer_idx, layer in enumerate(decoder_layers):
            layer_cache = layer_caches[layer_idx]

            self_attn = layer.self_attn
            cross_attn = layer.multihead_attn

            sa_input = layer.norm1(x)

            q_raw = self._project_q(self_attn, sa_input)
            k_new_raw, v_new_raw = self._project_kv(self_attn, sa_input)

            num_heads = self._decoder_num_heads[layer_idx]
            q = self._split_heads(q_raw, num_heads)
            k_new = self._split_heads(k_new_raw, num_heads)
            v_new = self._split_heads(v_new_raw, num_heads)

            self_k = layer_cache["self_k"]
            self_v = layer_cache["self_v"]
            k_cat = torch.cat([self_k, k_new], dim=2)
            v_cat = torch.cat([self_v, v_new], dim=2)

            sa_out = self._scaled_dot_product_attention(
                q=q,
                k=k_cat,
                v=v_cat,
                out_proj=self_attn.out_proj,
            )
            x = x + layer.dropout1(sa_out)

            ca_input = layer.norm2(x)
            q_cross_raw = self._project_q(cross_attn, ca_input)
            q_cross = self._split_heads(q_cross_raw, num_heads)

            mem_k = layer_cache["mem_k"]
            mem_v = layer_cache["mem_v"]

            ca_out = self._scaled_dot_product_attention(
                q=q_cross,
                k=mem_k,
                v=mem_v,
                out_proj=cross_attn.out_proj,
                key_padding_mask=memory_key_padding_mask,
            )
            x = x + layer.dropout2(ca_out)

            ff_input = layer.norm3(x)
            ff_out = layer.linear2(layer.dropout(layer.activation(layer.linear1(ff_input))))
            x = x + layer.dropout3(ff_out)

            layer_cache["self_k"] = k_cat
            layer_cache["self_v"] = v_cat

        if self._decoder_norm is not None:
            x = self._decoder_norm(x)

        logits_last = self.fc_out(x)[:, -1, :]
        cache["step"] = step + 1
        return logits_last, cache

    @torch.inference_mode()
    def generate(
        self,
        src: BatchedFeatures,
        src_lengths: FeatureLengths,
        *,
        sos_idx: int,
        eos_idx: int,
        max_len: int,
    ) -> BatchedTokens:
        """Greedily generate token sequences using KV-cache decoding."""

        batch_size = src.size(0)
        device = src.device

        memory, mem_mask = self.encode(src, src_lengths)
        kv_cache = self.init_kv_cache(memory)

        tgt = torch.full(
            (batch_size, max_len),
            fill_value=self.pad_idx,
            dtype=torch.long,
            device=device,
        )
        tgt[:, 0] = sos_idx
        unfinished_seqs = torch.ones(batch_size, dtype=torch.bool, device=device)

        for step in range(1, max_len):
            last_token = tgt[:, step - 1 : step]
            next_token_logits, kv_cache = self.decode_step(
                tgt_last=last_token,
                memory=memory,
                memory_key_padding_mask=mem_mask,
                cache=kv_cache,
            )

            next_token = torch.argmax(next_token_logits, dim=-1)

            tgt[:, step] = torch.where(unfinished_seqs, next_token, tgt[:, step])

            unfinished_seqs = unfinished_seqs & (next_token != eos_idx)

            if not unfinished_seqs.any():
                tgt = tgt[:, : step + 1]
                break

        return tgt

    def _export_encoder_forward(
        self, src: Tensor, src_lengths: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Custom forward for ONNX encoder tracing without dynamic control flows."""
        src_conv = src.transpose(1, 2)
        src_features = self.input_proj(src_conv).transpose(1, 2)
        src_emb = self.src_pe(src_features)

        conv1_len = self._conv1d_output_lengths(src_lengths, self.conv1)
        conv2_len = self._conv1d_output_lengths(conv1_len, self.conv2)
        downsampled_lengths = conv2_len.clamp_min(0)

        ones = torch.ones_like(src_features[:, :, 0], dtype=torch.long)
        steps = torch.cumsum(ones, dim=1) - 1
        mem_mask = steps >= downsampled_lengths.unsqueeze(1)

        x = src_emb
        for layer in self.transformer.encoder.layers:
            if not layer.norm_first:
                raise RuntimeError("ONNX encoder expects norm_first=True.")

            sa_input = layer.norm1(x)
            q_raw = self._project_q(layer.self_attn, sa_input)
            k_raw, v_raw = self._project_kv(layer.self_attn, sa_input)

            num_heads = layer.self_attn.num_heads
            q = self._split_heads(q_raw, num_heads)
            k = self._split_heads(k_raw, num_heads)
            v = self._split_heads(v_raw, num_heads)

            sa_out = self._scaled_dot_product_attention(
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

        if self.transformer.encoder.norm is not None:
            x = self.transformer.encoder.norm(x)

        memory = x
        mem_k = []
        mem_v = []
        for layer in self.transformer.decoder.layers:
            num_heads = layer.multihead_attn.num_heads
            mem_k_raw, mem_v_raw = self._project_kv(layer.multihead_attn, memory)
            mem_k.append(self._split_heads(mem_k_raw, num_heads))
            mem_v.append(self._split_heads(mem_v_raw, num_heads))

        return memory, mem_mask, torch.stack(mem_k, dim=0), torch.stack(mem_v, dim=0)

    def _export_decoder_step_forward(
        self,
        tgt_last: Tensor,
        mem_k: Tensor,
        mem_v: Tensor,
        mem_mask: Tensor,
        step: Tensor,
        self_k: Tensor,
        self_v: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Custom forward for ONNX decoder incremental step tracing."""
        x = self.tgt_tok_emb(tgt_last) * math.sqrt(self.d_model)

        step_long = step.to(dtype=torch.long).reshape(1)
        pe = self.tgt_pe.pe.squeeze(0)
        pe_step = torch.index_select(pe, dim=0, index=step_long).unsqueeze(0)
        x = x + pe_step
        x = self.tgt_pe.dropout(x)

        new_self_k = []
        new_self_v = []

        for layer_idx, layer in enumerate(self.transformer.decoder.layers):
            if not layer.norm_first:
                raise RuntimeError("ONNX decoder-step expects norm_first=True.")

            k_prev = self_k[layer_idx]
            v_prev = self_v[layer_idx]

            sa_input = layer.norm1(x)
            q_raw = self._project_q(layer.self_attn, sa_input)
            k_new_raw, v_new_raw = self._project_kv(layer.self_attn, sa_input)

            num_heads = layer.self_attn.num_heads
            q = self._split_heads(q_raw, num_heads)
            k_new = self._split_heads(k_new_raw, num_heads)
            v_new = self._split_heads(v_new_raw, num_heads)

            k_cat = torch.cat([k_prev, k_new], dim=2)
            v_cat = torch.cat([v_prev, v_new], dim=2)

            sa_out = self._scaled_dot_product_attention(
                q=q,
                k=k_cat,
                v=v_cat,
                out_proj=layer.self_attn.out_proj,
            )
            x = x + layer.dropout1(sa_out)

            ca_input = layer.norm2(x)
            q_cross_raw = self._project_q(layer.multihead_attn, ca_input)
            q_cross = self._split_heads(q_cross_raw, num_heads)

            ca_out = self._scaled_dot_product_attention(
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

        if self.transformer.decoder.norm is not None:
            x = self.transformer.decoder.norm(x)

        logits = self.fc_out(x)[:, -1, :]
        return logits, torch.stack(new_self_k, dim=0), torch.stack(new_self_v, dim=0)

    def get_onnx_export_configs(self, device: str = "cpu") -> list[OnnxExportConfiguration]:
        """Return configurations allowing DynamicOnnxExportWrapper to wrap inner logic."""
        dummy_src_len = 64
        dummy_tgt_len = 1
        dummy_cache_len = 1
        in_channels = self.conv1.in_channels

        src = torch.zeros((1, dummy_src_len, in_channels), dtype=torch.float32, device=device)
        src_lengths = torch.tensor([dummy_src_len], dtype=torch.long, device=device)
        tgt_last = torch.full((1, dummy_tgt_len), self.pad_idx, dtype=torch.long, device=device)

        step = torch.tensor([0], dtype=torch.long, device=device)

        num_layers = len(self.transformer.decoder.layers)
        num_heads = self.transformer.decoder.layers[0].self_attn.num_heads
        head_dim = self.d_model // num_heads

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

        with torch.no_grad():
            memory, mem_mask, mem_k, mem_v = self._export_encoder_forward(src, src_lengths)

        enc_config = OnnxExportConfiguration(
            name="encoder",
            export_fun=self._export_encoder_forward,
            dummy_inputs=(src, src_lengths),
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
        )

        dec_config = OnnxExportConfiguration(
            name="decoder_step",
            export_fun=self._export_decoder_step_forward,
            dummy_inputs=(tgt_last, mem_k, mem_v, mem_mask, step, self_k, self_v),
            input_names=["tgt_last", "mem_k", "mem_v", "mem_mask", "step", "self_k", "self_v"],
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
        )

        return [enc_config, dec_config]

    @classmethod
    def run_onnx_inference(
        cls,
        sessions,
        src_features: "BatchedFeatures",
        src_lengths: "FeatureLengths",
        vocab: "LatexVocab",
        max_len: int,
    ) -> list[str]:
        """Using provided onnx-runtime sessions perform inference on `src_features`."""

        encoder_session = sessions["encoder"]
        decoder_session = sessions["decoder_step"]

        src = np.array(src_features)
        lengths = np.array(src_lengths)

        batch_size = src.shape[0]

        enc_inputs = {
            "src": src.astype(np.float32),
            "src_lengths": lengths.astype(np.int64),
        }
        memory, mem_mask, mem_k, mem_v = encoder_session.run(None, enc_inputs)

        num_layers = mem_k.shape[0]
        num_heads = mem_k.shape[2]
        head_dim = mem_k.shape[4]

        self_k = np.zeros((num_layers, batch_size, num_heads, 0, head_dim), dtype=np.float32)
        self_v = np.zeros((num_layers, batch_size, num_heads, 0, head_dim), dtype=np.float32)

        tgt = np.full((batch_size, max_len), fill_value=vocab.PAD, dtype=np.int64)
        tgt[:, 0] = vocab.SOS

        unfinished_seqs = np.ones(batch_size, dtype=bool)

        step = np.array([0], dtype=np.int64)

        for i in range(1, max_len):
            tgt_last = tgt[:, i - 1 : i]

            dec_inputs = {
                "tgt_last": tgt_last,
                "mem_k": mem_k,
                "mem_v": mem_v,
                "mem_mask": mem_mask,
                "step": step,
                "self_k": self_k,
                "self_v": self_v,
            }

            logits, self_k, self_v = decoder_session.run(None, dec_inputs)

            next_token = np.argmax(logits, axis=-1)
            tgt[:, i] = np.where(unfinished_seqs, next_token, tgt[:, i])

            unfinished_seqs = unfinished_seqs & (next_token != vocab.EOS)

            if not np.any(unfinished_seqs):
                break

            step[0] += 1

        results = []
        for seq in tgt:
            clean_seq = []
            for token in seq:
                match t := int(token):
                    case vocab.EOS:
                        clean_seq.append(t)
                        break
                    case vocab.PAD:
                        continue
                    case _:
                        clean_seq.append(t)
            print(clean_seq)
            results.append(vocab.decode_sequence(clean_seq))

        return results
