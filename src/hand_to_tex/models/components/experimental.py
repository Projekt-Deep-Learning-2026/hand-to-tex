import math

import torch
import torch.nn as nn
from torch import Tensor


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence-first transformer inputs.

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
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

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
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.fc_out(out)
