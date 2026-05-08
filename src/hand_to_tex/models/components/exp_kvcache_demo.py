import math

import torch
import torch.nn as nn
from torch import Tensor

from hand_to_tex.models.components.experimental import ExperimentalTransformer
from hand_to_tex.types import (
    BatchedFeatures,
    BatchedTokens,
    DecoderKVCache,
    FeatureLengths,
    LayerKVCache,
    TensorBool,
)


class ExperimentalTransformerKVCacheDemo(ExperimentalTransformer):
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
