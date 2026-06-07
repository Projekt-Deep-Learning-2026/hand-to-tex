"""Tests for the ExperimentalTransformer model.

Validates forward pass, encoder/decoder separation, masking, and gradient flow.
"""

from __future__ import annotations

import pytest
import torch

from hand_to_tex.models.components import ExperimentalTransformer


def _build_model(vocab_size: int = 32, pad_idx: int = 0) -> ExperimentalTransformer:
    """Build a tiny transformer for fast unit testing.

    Args:
        vocab_size: Output vocabulary size
        pad_idx: Padding token index

    Returns:
        ExperimentalTransformer with minimal dimensions
    """
    return ExperimentalTransformer(
        in_channels=12,
        vocab_size=vocab_size,
        pad_idx=pad_idx,
        d_model=32,
        nhead=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=64,
        dropout=0.0,
    )


class TestForwardPass:
    """Test the main forward pass."""

    def test_forward_basic_functionality(self) -> None:
        """Forward pass must produce logits of correct shape and dtype."""
        torch.manual_seed(0)
        vocab_size = 32
        model = _build_model(vocab_size=vocab_size).eval()

        B, T_src, T_tgt = 2, 24, 5
        src = torch.randn(B, T_src, 12)
        src_lengths = torch.tensor([T_src, T_src - 4], dtype=torch.long)
        tgt = torch.randint(low=1, high=vocab_size, size=(B, T_tgt))

        out = model(src=src, src_lengths=src_lengths, tgt=tgt)

        assert out.shape == (B, T_tgt, vocab_size)
        assert out.dtype == torch.float32
        assert torch.isfinite(out).all()

    def test_forward_deterministic_eval_mode(self) -> None:
        """Eval mode must be deterministic."""
        torch.manual_seed(0)
        model = _build_model().eval()
        src = torch.randn(2, 24, 12)
        src_lengths = torch.tensor([24, 20], dtype=torch.long)
        tgt = torch.randint(low=1, high=10, size=(2, 5))

        out1 = model(src=src, src_lengths=src_lengths, tgt=tgt)
        out2 = model(src=src, src_lengths=src_lengths, tgt=tgt)

        assert torch.allclose(out1, out2)

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
    def test_forward_various_batch_sizes(self, batch_size: int) -> None:
        """Forward must work with different batch sizes."""
        torch.manual_seed(0)
        model = _build_model().eval()

        src = torch.randn(batch_size, 24, 12)
        src_lengths = torch.tensor([24] * batch_size, dtype=torch.long)
        tgt = torch.randint(low=1, high=32, size=(batch_size, 5))

        out = model(src=src, src_lengths=src_lengths, tgt=tgt)
        assert out.shape == (batch_size, 5, 32)

    def test_forward_causal_masking(self) -> None:
        """Causal mask must prevent future positions from affecting past positions.

        Changes in target positions > 3 should not affect outputs at positions <= 3.
        """
        torch.manual_seed(0)
        pad_idx = 0
        model = _build_model(pad_idx=pad_idx).eval()

        src = torch.randn(2, 24, 12)
        src_lengths = torch.tensor([24, 24], dtype=torch.long)

        tgt_a = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
        tgt_b = torch.tensor([[1, 2, 3, 7, 8], [1, 2, 3, 7, 8]])

        out_a = model(src=src, src_lengths=src_lengths, tgt=tgt_a)
        out_b = model(src=src, src_lengths=src_lengths, tgt=tgt_b)

        # Early positions should not change
        assert torch.allclose(out_a[:, :3, :], out_b[:, :3, :], atol=1e-6)
        # Later positions should change
        assert not torch.allclose(out_a[:, 3:, :], out_b[:, 3:, :], atol=1e-6)


class TestEncodingDecoding:
    """Test separate encode and decode paths."""

    def test_encode_output_structure(self) -> None:
        """Encode must return memory and padding mask."""
        torch.manual_seed(0)
        model = _build_model().eval()
        B, T_src = 2, 24
        src = torch.randn(B, T_src, 12)
        src_lengths = torch.tensor([T_src, T_src - 8], dtype=torch.long)

        memory, mem_mask = model.encode(src, src_lengths)

        assert memory.ndim == 3
        assert memory.shape[0] == B
        assert memory.shape[2] == model.d_model
        assert mem_mask.shape == (B, memory.shape[1])
        assert mem_mask.dtype == torch.bool

    def test_decode_output_structure(self) -> None:
        """Decode must return logits."""
        torch.manual_seed(0)
        vocab_size = 32
        model = _build_model(vocab_size=vocab_size).eval()

        B, T_src = 2, 24
        src = torch.randn(B, T_src, 12)
        src_lengths = torch.tensor([T_src, T_src], dtype=torch.long)
        memory, mem_mask = model.encode(src, src_lengths)

        tgt = torch.randint(low=1, high=vocab_size, size=(B, 4))
        out = model.decode(tgt=tgt, memory=memory, memory_key_padding_mask=mem_mask)

        assert out.shape == (B, 4, vocab_size)
        assert torch.isfinite(out).all()

    def test_encode_decode_matches_forward(self) -> None:
        """Separate encode+decode must match forward pass."""
        torch.manual_seed(0)
        model = _build_model().eval()

        B, T_src, T_tgt = 2, 24, 5
        src = torch.randn(B, T_src, 12)
        src_lengths = torch.tensor([T_src, T_src], dtype=torch.long)
        tgt = torch.randint(low=1, high=32, size=(B, T_tgt))

        # Full forward
        out_full = model(src=src, src_lengths=src_lengths, tgt=tgt)

        # Separate encode/decode
        memory, mem_mask = model.encode(src, src_lengths)
        out_separate = model.decode(tgt=tgt, memory=memory, memory_key_padding_mask=mem_mask)

        assert torch.allclose(out_full, out_separate, atol=1e-6)


class TestBackwardPass:
    """Test gradient computation."""

    def test_backward_all_params_receive_gradients(self) -> None:
        """All learnable parameters must receive gradients."""
        torch.manual_seed(0)
        vocab_size = 32
        model = _build_model(vocab_size=vocab_size).train()

        B = 2
        src = torch.randn(B, 24, 12)
        src_lengths = torch.tensor([24, 24], dtype=torch.long)
        tgt = torch.randint(low=1, high=vocab_size, size=(B, 5))

        out = model(src=src, src_lengths=src_lengths, tgt=tgt)
        loss = out.mean()
        loss.backward()

        missing_grads = [
            name for name, p in model.named_parameters() if p.requires_grad and p.grad is None
        ]
        assert not missing_grads, f"Parameters with no gradient: {missing_grads}"

    def test_backward_gradients_are_finite(self) -> None:
        """All gradients must be finite."""
        torch.manual_seed(0)
        model = _build_model().train()

        src = torch.randn(2, 24, 12)
        src_lengths = torch.tensor([24, 24], dtype=torch.long)
        tgt = torch.randint(low=1, high=32, size=(2, 5))

        out = model(src=src, src_lengths=src_lengths, tgt=tgt)
        loss = out.mean()
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), f"Non-finite grad in {name}"
