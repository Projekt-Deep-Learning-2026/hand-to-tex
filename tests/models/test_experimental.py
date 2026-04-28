from __future__ import annotations

import torch

from hand_to_tex.models.components import ExperimentalTransformer


def _build_tiny_model(vocab_size: int = 32, pad_idx: int = 0) -> ExperimentalTransformer:
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


class TestExperimentalTransformerForward:
    def test_forward_returns_logits_with_expected_shape(self):
        torch.manual_seed(0)
        vocab_size = 32
        model = _build_tiny_model(vocab_size=vocab_size).eval()

        B, T_src, T_tgt = 2, 24, 5
        src = torch.randn(B, T_src, 12)
        src_lengths = torch.tensor([T_src, T_src - 4], dtype=torch.long)
        tgt = torch.randint(low=1, high=vocab_size, size=(B, T_tgt))

        out = model(src=src, src_lengths=src_lengths, tgt=tgt)

        assert out.shape == (B, T_tgt, vocab_size)
        assert out.dtype == torch.float32
        assert torch.isfinite(out).all()

    def test_forward_handles_minimum_sized_batch(self):
        torch.manual_seed(0)
        model = _build_tiny_model().eval()
        B = 2
        # T_src=24 -> conv1(k=5,s=2,p=1): floor((24+2-4-1)/2)+1 = 11
        #          -> conv2(k=5,s=2,p=2): floor((11+4-4-1)/2)+1 = 6
        T_src = 24
        src = torch.randn(B, T_src, 12)
        src_lengths = torch.tensor([T_src, T_src], dtype=torch.long)
        tgt = torch.randint(low=1, high=10, size=(B, 4))

        out = model(src=src, src_lengths=src_lengths, tgt=tgt)
        assert out.shape[0] == B
        assert out.shape[1] == tgt.shape[1]

    def test_forward_is_deterministic_in_eval_mode(self):
        torch.manual_seed(0)
        model = _build_tiny_model().eval()
        src = torch.randn(2, 24, 12)
        src_lengths = torch.tensor([24, 20], dtype=torch.long)
        tgt = torch.randint(low=1, high=10, size=(2, 5))

        out1 = model(src=src, src_lengths=src_lengths, tgt=tgt)
        out2 = model(src=src, src_lengths=src_lengths, tgt=tgt)
        assert torch.allclose(out1, out2)

    def test_padding_in_target_does_not_change_unpadded_positions(self):
        """Causal mask must prevent earlier positions from seeing later tokens."""
        torch.manual_seed(0)
        pad_idx = 0
        model = _build_tiny_model(pad_idx=pad_idx).eval()

        src = torch.randn(2, 24, 12)
        src_lengths = torch.tensor([24, 24], dtype=torch.long)

        tgt_a = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
        tgt_b = torch.tensor([[1, 2, 3, 7, 8], [1, 2, 3, 7, 8]])

        out_a = model(src=src, src_lengths=src_lengths, tgt=tgt_a)
        out_b = model(src=src, src_lengths=src_lengths, tgt=tgt_b)
        assert torch.allclose(out_a[:, :3, :], out_b[:, :3, :], atol=1e-6)
        assert not torch.allclose(out_a[:, 3:, :], out_b[:, 3:, :], atol=1e-6)


class TestEncodeDecode:
    def test_encode_returns_memory_and_padding_mask(self):
        torch.manual_seed(0)
        model = _build_tiny_model().eval()
        B, T_src = 2, 24
        src = torch.randn(B, T_src, 12)
        src_lengths = torch.tensor([T_src, T_src - 8], dtype=torch.long)

        memory, mem_mask = model.encode(src, src_lengths)

        assert memory.ndim == 3
        assert memory.shape[0] == B
        assert memory.shape[2] == model.d_model
        assert mem_mask.shape == (B, memory.shape[1])
        assert mem_mask.dtype == torch.bool

    def test_decode_returns_logits(self):
        torch.manual_seed(0)
        vocab_size = 32
        model = _build_tiny_model(vocab_size=vocab_size).eval()

        B, T_src = 2, 24
        src = torch.randn(B, T_src, 12)
        src_lengths = torch.tensor([T_src, T_src], dtype=torch.long)
        memory, mem_mask = model.encode(src, src_lengths)

        tgt = torch.randint(low=1, high=vocab_size, size=(B, 4))
        out = model.decode(tgt=tgt, memory=memory, memory_key_padding_mask=mem_mask)

        assert out.shape == (B, 4, vocab_size)
        assert torch.isfinite(out).all()


class TestPaddingMaskAndLengthMath:
    def test_padding_mask_marks_pad_positions_true(self):
        model = _build_tiny_model()
        lengths = torch.tensor([3, 5], dtype=torch.long)
        mask = model._get_padding_mask(lengths, max_len=6)

        expected = torch.tensor(
            [
                [False, False, False, True, True, True],
                [False, False, False, False, False, True],
            ]
        )
        assert torch.equal(mask, expected)

    def test_downsampled_lengths_match_actual_conv_output(self):
        model = _build_tiny_model().eval()
        T_src = 32
        src = torch.randn(2, T_src, 12)
        x = src.transpose(1, 2)
        x = model.input_proj(x)
        actual_len = x.shape[2]

        predicted = model._calc_downsampled_lengths(torch.tensor([T_src, T_src], dtype=torch.long))
        assert int(predicted[0].item()) == actual_len
        assert int(predicted[1].item()) == actual_len

    def test_downsampled_lengths_clamp_at_zero(self):
        model = _build_tiny_model()
        out = model._calc_downsampled_lengths(torch.tensor([0, 1, 2], dtype=torch.long))
        assert (out >= 0).all()


class TestBackwardPass:
    def test_backward_produces_gradients_for_all_params(self):
        """Every learnable parameter must receive a gradient — a disconnected param is a wiring bug."""
        torch.manual_seed(0)
        vocab_size = 32
        model = _build_tiny_model(vocab_size=vocab_size).train()

        B = 2
        src = torch.randn(B, 24, 12)
        src_lengths = torch.tensor([24, 24], dtype=torch.long)
        tgt = torch.randint(low=1, high=vocab_size, size=(B, 5))

        out = model(src=src, src_lengths=src_lengths, tgt=tgt)
        loss = out.mean()
        loss.backward()

        missing = [
            name for name, p in model.named_parameters() if p.requires_grad and p.grad is None
        ]
        assert missing == [], f"Parameters with no gradient: {missing}"
