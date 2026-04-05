import math

import torch
import torch.nn as nn
from torch import Tensor


class PositionalEncoding(nn.Module):
    """
    Injects positional information into the input representations.

    This module uses sinusoidal and cosinusoidal functions of different frequencies
    to encode the absolute position of a token or feature sequence. The resulting
    positional embeddings are added directly to the input tensor, allowing the
    Transformer to possess a sense of sequence order, which it inherently lacks.

    Attributes
    ----------
    dropout : nn.Dropout
        Dropout layer applied to the final output to prevent overfitting.
    pe : Tensor
        A registered buffer containing the pre-computed positional encodings
        up to the specified maximum sequence length.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 10000):
        """
        Initializes the PositionalEncoding module.

        Parameters
        ----------
        d_model : int
            The dimensionality of the embeddings to which positional encodings will be added.
        dropout : float, optional
            The dropout probability. Default is 0.1.
        max_len : int, optional
            The maximum possible sequence length for which to pre-compute the encodings.
            Default is 10000.
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
        """
        Applies positional encoding to the input tensor.

        Parameters
        ----------
        x : Tensor
            The input tensor of shape (batch_size, sequence_length, embedding_dim).

        Returns
        -------
        Tensor
            The input tensor with added positional encodings and applied dropout,
            matching the shape of the original input.
        """
        x = x + self.pe[:, : x.size(1)]  # type: ignore
        return self.dropout(x)


class BaselineTransformer(nn.Module):
    """
    Sequence-to-sequence Baseline Transformer Model with Conv1D downsampling.

    This architecture is designed to map continuous online handwriting features
    into discrete LaTeX formula tokens. It utilizes a 1D Convolutional Neural Network
    layer to effectively reduce the sequence length and computational complexity of
    long physical traces, followed by a standard Transformer Encoder-Decoder block
    to handle the overarching sequence translation process.
    """

    def __init__(
        self,
        vocab_size: int,
        pad_idx: int,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        in_features: int = 10,
    ):
        """
        Initializes the BaselineTransformer architecture.

        Parameters
        ----------
        vocab_size : int
            The total number of available LaTeX tokens in the output vocabulary.
        pad_idx : int
            The integer index representing the padding token in the vocabulary.
        d_model : int, optional
            The dimensionality of the intermediate embeddings and attention mechanisms.
        nhead : int, optional
            The number of parallel attention heads in the Multi-Head Attention modules.
        num_encoder_layers : int, optional
            The number of stacked Transformer Encoder layers.
        num_decoder_layers : int, optional
            The number of stacked Transformer Decoder layers.
        dim_feedforward : int, optional
            The dimensionality of the hidden layer inside the feedforward networks
            within the Transformer layers.
        dropout : float, optional
            The dropout probability applied across the model.
        in_features : int, optional
            The number of distinct features provided per chronological point in the
            input tensor. Default is 10.
        """
        super().__init__()
        self.pad_idx = pad_idx
        self.d_model = d_model

        self.input_conv = nn.Conv1d(
            in_channels=in_features,
            out_channels=d_model,
            kernel_size=5,
            stride=2,
            padding=2,
        )
        self.input_pe = PositionalEncoding(d_model, dropout)

        self.target_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.target_pe = PositionalEncoding(d_model, dropout)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        if hasattr(self.transformer.encoder, "enable_nested_tensor"):
            self.transformer.encoder.enable_nested_tensor = False

        self.out_proj = nn.Linear(d_model, vocab_size)

    def generate_square_subsequent_mask(self, sz: int, device: torch.device) -> Tensor:
        """
        Generates a causal upper-triangular masking matrix for the Transformer Decoder.
        """
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, 0.0)
        return mask

    def forward(self, src: Tensor, src_lengths: Tensor, tgt: Tensor) -> Tensor:
        """
        Orchestrates the complete forward pass through the sequence mapping pipeline.

        The raw input sequence is downsampled via the convolutional block, enriched
        with positional encodings, and finally routed through the Transformer
        Encoder and Decoder along with dynamically generated key padding masks
        and causal autoregressive target masks.

        Parameters
        ----------
        src : Tensor
            The input tensor representing dynamic handwriting features, formatted with
            shape (batch_size, feature_sequence_length, in_features).
        src_lengths : list[int]
            A list containing the actual unpadded temporal lengths for every sample in the batch.
        tgt : Tensor
            The target tensor containing integer token indices from the LaTeX vocabulary,
            formatted with shape (batch_size, token_sequence_length).

        Returns
        -------
        Tensor
            An unnormalized logit tensor of shape (batch_size, token_sequence_length, vocab_size),
            representing the probability distribution over the available LaTeX vocabulary tokens.
        """
        B, _T_feat, _ = src.shape
        T_tok = tgt.shape[1]

        src_conv = src.permute(0, 2, 1)
        src_encoded = self.input_conv(src_conv)
        src_encoded = src_encoded.permute(0, 2, 1)

        new_src_lens = (
            (src_lengths + 2 * self.input_conv.padding[0] - self.input_conv.kernel_size[0])  # type: ignore
            // self.input_conv.stride[0]
        ) + 1

        max_len_conv = src_encoded.shape[1]
        src_key_padding_mask = torch.arange(max_len_conv, device=src.device).unsqueeze(0).expand(
            B, max_len_conv
        ) >= new_src_lens.unsqueeze(1)

        src_encoded = self.input_pe(src_encoded)

        tgt_embed = self.target_embed(tgt) * math.sqrt(self.d_model)
        tgt_embed = self.target_pe(tgt_embed)

        src_key_padding_mask = torch.arange(max_len_conv, device=src.device).unsqueeze(0).expand(
            B, max_len_conv
        ) >= new_src_lens.unsqueeze(1)
        tgt_key_padding_mask = tgt == self.pad_idx

        tgt_mask = self.generate_square_subsequent_mask(T_tok, src.device)

        output = self.transformer(
            src=src_encoded,
            tgt=tgt_embed,
            src_mask=None,
            tgt_mask=tgt_mask,
            memory_mask=None,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )

        return self.out_proj(output)
