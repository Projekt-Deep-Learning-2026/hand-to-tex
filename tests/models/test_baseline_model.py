from __future__ import annotations

import torch

from hand_to_tex.models.components.baseline_model import (
    BaselineTransformer,
    PositionalEncoding,
)


class TestBaselinePositionalEncoding:
    def test_output_shape_matches_input(self):
        torch.manual_seed(0)
        pe = PositionalEncoding(d_model=16, dropout=0.0)
        pe.eval()  # disable dropout for deterministic output
        x = torch.zeros(2, 7, 16)
        out = pe(x)
        assert out.shape == x.shape

    def test_encoding_is_added_not_replaced(self):
        torch.manual_seed(0)
        pe = PositionalEncoding(d_model=16, dropout=0.0)
        pe.eval()
        x = torch.ones(1, 5, 16)
        out = pe(x)
        assert not torch.allclose(out, x)


class TestBaselineTransformerForward:
    def _build(self, vocab_size: int = 32, pad_idx: int = 0) -> BaselineTransformer:
        return BaselineTransformer(
            vocab_size=vocab_size,
            pad_idx=pad_idx,
            d_model=32,
            nhead=2,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=64,
            dropout=0.0,
        )

    def test_forward_returns_logits(self):
        torch.manual_seed(0)
        vocab_size = 32
        model = self._build(vocab_size=vocab_size).eval()

        B, T_src, T_tgt = 2, 24, 5
        src = torch.randn(B, T_src, 12)
        src_lengths = torch.tensor([T_src, T_src - 4], dtype=torch.long)
        tgt = torch.randint(low=1, high=vocab_size, size=(B, T_tgt))

        out = model(src=src, src_lengths=src_lengths, tgt=tgt)

        assert out.shape == (B, T_tgt, vocab_size)
        assert torch.isfinite(out).all()

    def test_causal_mask_has_correct_shape(self):
        model = self._build()
        mask = model.generate_square_subsequent_mask(5, device=torch.device("cpu"))
        assert mask.shape == (5, 5)
        assert mask[0, 1].item() == float("-inf")
        assert mask[1, 0].item() == 0.0
