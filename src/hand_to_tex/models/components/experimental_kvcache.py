import math

import torch
import torch.nn as nn
from torch import Tensor

type LayerKVCache = dict[str, Tensor]
type DecoderKVCache = dict[str, object]


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for batch-first transformer inputs.

    The module expects tensors shaped `(batch, seq_len, d_model)` and adds
    deterministic sinusoidal vectors to each position before dropout.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """Precompute sinusoidal encodings and register them as a buffer.

        Parameters
        ----------
        d_model:
            Embedding size of the model.
        dropout:
            Dropout probability applied after positional addition.
        max_len:
            Maximum sequence length covered by the precomputed table.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """Add position encodings to an input batch.

        Parameters
        ----------
        x:
            Input tensor of shape `(B, T, D)`.

        Returns
        -------
        Tensor
            Tensor with positional information added, same shape as input.
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)

    def forward_step(self, x: Tensor, step: int) -> Tensor:
        """Add position encoding for a single decoding step.

        Parameters
        ----------
        x:
            Input tensor of shape `(B, 1, D)`.
        step:
            Zero-based decoding step.

        Returns
        -------
        Tensor
            Positionalized tensor of shape `(B, 1, D)`.
        """
        x = x + self.pe[:, step : step + 1, :]
        return self.dropout(x)


class ExperimentalTransformer(nn.Module):
    """Sequence-to-sequence transformer for handwriting feature decoding.

    Source inputs are expected as `(B, T_src, in_channels)` float features.
    Target inputs are `(B, T_tgt)` token ids used for teacher forcing.
    """

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
        """Initialize encoder/decoder stack and projection heads.

        Parameters
        ----------
        in_channels:
            Number of per-timestep source features.
        vocab_size:
            Size of output token vocabulary.
        pad_idx:
            Vocabulary index used for padding tokens.
        d_model:
            Hidden embedding size used throughout the transformer.
        nhead:
            Number of attention heads.
        num_encoder_layers:
            Number of transformer encoder blocks.
        num_decoder_layers:
            Number of transformer decoder blocks.
        dim_feedforward:
            Feed-forward width in transformer blocks.
        dropout:
            Dropout probability used in transformer and positional layers.
        """
        super().__init__()
        self.pad_idx = pad_idx
        self.d_model = d_model

        self.conv1 = nn.Conv1d(in_channels, d_model, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=5, stride=2, padding=2)

        self.input_proj = nn.Sequential(
            self.conv1,
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            self.conv2,
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )

        self.tgt_tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)

        self.src_pe = PositionalEncoding(d_model, dropout)
        self.tgt_pe = PositionalEncoding(d_model, dropout)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.fc_out = nn.Linear(d_model, vocab_size)

    def _get_padding_mask(self, lengths: Tensor, max_len: int) -> Tensor:
        """Build a boolean key-padding mask from sequence lengths.

        Parameters
        ----------
        lengths:
            Tensor `(B,)` with valid sequence lengths.
        max_len:
            Maximum sequence length in the padded batch.

        Returns
        -------
        Tensor
            Boolean mask `(B, max_len)` where `True` marks padding positions.
        """
        steps = torch.arange(max_len, device=lengths.device).unsqueeze(0)
        return steps >= lengths.unsqueeze(1)

    @staticmethod
    def _conv1d_output_lengths(lengths: Tensor, conv: nn.Conv1d) -> Tensor:
        """Compute output lengths after applying a Conv1d layer.

        Uses the exact Conv1d formula:
        `L_out = floor((L_in + 2P - D*(K-1) - 1) / S) + 1`.
        """
        kernel_size = conv.kernel_size[0]
        stride = conv.stride[0]
        padding = conv.padding[0]
        dilation = conv.dilation[0]

        numerator = lengths + 2 * padding - dilation * (kernel_size - 1) - 1
        return torch.div(numerator, stride, rounding_mode="floor") + 1

    def _calc_downsampled_lengths(self, lengths: Tensor) -> Tensor:
        """Compute source lengths after the two strided projection convolutions.

        Returns
        -------
        Tensor
            Tensor `(B,)` with projected source lengths.
        """
        conv1_len = self._conv1d_output_lengths(lengths, self.conv1)
        conv2_len = self._conv1d_output_lengths(conv1_len, self.conv2)
        return conv2_len.clamp_min(0)

    @staticmethod
    def _build_causal_mask(seq_len: int, device: torch.device) -> Tensor:
        """Create a decoder causal mask where future positions are blocked."""
        return nn.Transformer.generate_square_subsequent_mask(seq_len, device=device).bool()

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
        key_padding_mask: Tensor | None = None,
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

    def forward(self, src: Tensor, src_lengths: Tensor, tgt: Tensor) -> Tensor:
        """Run the full teacher-forced transformer pass.

        Parameters
        ----------
        src:
            Source feature tensor `(B, T_src, C)`.
        src_lengths:
            Valid source lengths `(B,)` before convolutional downsampling.
        tgt:
            Target token ids `(B, T_tgt)` used as decoder input.

        Returns
        -------
        Tensor
            Decoder logits `(B, T_tgt, vocab_size)`.
        """
        src_conv = src.transpose(1, 2)
        src_features = self.input_proj(src_conv).transpose(1, 2)

        src_emb = self.src_pe(src_features)

        new_src_lengths = self._calc_downsampled_lengths(src_lengths)
        src_key_padding_mask = self._get_padding_mask(new_src_lengths, src_features.size(1))

        tgt_emb = self.tgt_pe(self.tgt_tok_emb(tgt) * math.sqrt(self.d_model))
        tgt_key_padding_mask = tgt == self.pad_idx

        tgt_seq_len = tgt.size(1)
        tgt_causal_mask = self._build_causal_mask(tgt_seq_len, tgt.device)

        output = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            src_mask=None,
            tgt_mask=tgt_causal_mask,
            tgt_is_causal=True,
            memory_mask=None,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )

        return self.fc_out(output)

    def encode(self, src: Tensor, src_lengths: Tensor) -> tuple[Tensor, Tensor]:
        """Encode source features into memory for autoregressive decoding.

        Parameters
        ----------
        src:
            Source feature tensor `(B, T_src, C)`.
        src_lengths:
            Valid source lengths `(B,)` before convolutional downsampling.

        Returns
        -------
        tuple[Tensor, Tensor]
            - `memory`: encoder output `(B, T_src', D)`
            - `src_key_padding_mask`: boolean mask `(B, T_src')`
        """
        src_conv = src.transpose(1, 2)

        src_features = self.input_proj(src_conv).transpose(1, 2)

        new_src_lengths = self._calc_downsampled_lengths(src_lengths)

        src_emb = self.src_pe(src_features)

        src_key_padding_mask = self._get_padding_mask(new_src_lengths, src_features.size(1))

        memory = self.transformer.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)

        return memory, src_key_padding_mask

    def decode(self, tgt: Tensor, memory: Tensor, memory_key_padding_mask: Tensor) -> Tensor:
        """Decode a target prefix given encoder memory.

        Parameters
        ----------
        tgt:
            Decoder input token ids `(B, T_tgt)`.
        memory:
            Encoder memory `(B, T_src', D)`.
        memory_key_padding_mask:
            Boolean mask `(B, T_src')` marking padded memory positions.

        Returns
        -------
        Tensor
            Decoder logits `(B, T_tgt, vocab_size)`.
        """

        tgt_emb = self.tgt_pe(self.tgt_tok_emb(tgt) * math.sqrt(self.d_model))

        tgt_causal_mask = self._build_causal_mask(tgt.size(1), tgt.device)
        tgt_key_padding_mask = tgt == self.pad_idx

        out = self.transformer.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_causal_mask,
            tgt_is_causal=True,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.fc_out(out)

    @torch.no_grad()
    def init_kv_cache(self, memory: Tensor) -> DecoderKVCache:
        """Initialize decoder cache for incremental generation.

        Parameters
        ----------
        memory:
            Encoder memory `(B, T_src', D)`.

        Returns
        -------
        DecoderKVCache
            Cache dictionary with current step and per-layer K/V tensors.
        """
        batch_size, _src_len, _ = memory.shape
        decoder_layers = self.transformer.decoder.layers
        layer_caches: list[LayerKVCache] = []

        for layer in decoder_layers:
            num_heads = layer.self_attn.num_heads
            head_dim = self.d_model // num_heads

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
        tgt_last: Tensor,
        memory: Tensor,
        memory_key_padding_mask: Tensor,
        cache: DecoderKVCache,
    ) -> tuple[Tensor, DecoderKVCache]:
        """Decode a single autoregressive step using cached projected K/V tensors.

        Parameters
        ----------
        tgt_last:
            Last decoder token ids `(B, 1)`.
        memory:
            Encoder memory `(B, T_src', D)`.
        memory_key_padding_mask:
            Boolean mask `(B, T_src')` marking padded memory positions.
        cache:
            Decoder cache returned by :meth:`init_kv_cache`.

        Returns
        -------
        tuple[Tensor, DecoderKVCache]
            - `logits_last`: logits for current step `(B, vocab_size)`
            - updated cache
        """
        _ = memory
        if tgt_last.dim() != 2 or tgt_last.size(1) != 1:
            raise ValueError("`tgt_last` must have shape (B, 1) for incremental decoding.")

        step = int(cache["step"])
        layer_caches = cache["layers"]
        if not isinstance(layer_caches, list):
            raise ValueError("Cache `layers` entry is malformed.")

        x = self.tgt_tok_emb(tgt_last) * math.sqrt(self.d_model)
        x = self.tgt_pe.forward_step(x, step)

        decoder_layers = self.transformer.decoder.layers

        for layer_idx, layer in enumerate(decoder_layers):
            if not layer.norm_first:
                raise RuntimeError(
                    "KV-cache decode_step currently supports `norm_first=True` decoder layers only."
                )

            layer_cache = layer_caches[layer_idx]
            if not isinstance(layer_cache, dict):
                raise ValueError(f"Malformed cache entry for decoder layer {layer_idx}.")

            sa_input = layer.norm1(x)

            q_raw = self._project_q(layer.self_attn, sa_input)
            k_new_raw, v_new_raw = self._project_kv(layer.self_attn, sa_input)

            num_heads = layer.self_attn.num_heads
            q = self._split_heads(q_raw, num_heads)
            k_new = self._split_heads(k_new_raw, num_heads)
            v_new = self._split_heads(v_new_raw, num_heads)

            k_prev = layer_cache["self_k"]
            v_prev = layer_cache["self_v"]
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

            mem_k = layer_cache["mem_k"]
            mem_v = layer_cache["mem_v"]

            ca_out = self._scaled_dot_product_attention(
                q=q_cross,
                k=mem_k,
                v=mem_v,
                out_proj=layer.multihead_attn.out_proj,
                key_padding_mask=memory_key_padding_mask,
            )
            x = x + layer.dropout2(ca_out)

            ff_input = layer.norm3(x)
            ff_out = layer.linear2(layer.dropout(layer.activation(layer.linear1(ff_input))))
            x = x + layer.dropout3(ff_out)

            layer_cache["self_k"] = k_cat
            layer_cache["self_v"] = v_cat

        if self.transformer.decoder.norm is not None:
            x = self.transformer.decoder.norm(x)

        logits_last = self.fc_out(x)[:, -1, :]
        cache["step"] = step + 1
        return logits_last, cache
