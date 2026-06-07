"""Tests for KV-cache implementations.

Verifies parity between standard decoding and KV-cache incremental decoding.
Ensures that optimized decoding paths yield identical results to teacher-forced
or standard autoregressive paths.
"""

from __future__ import annotations

import pytest
import torch

from hand_to_tex.models.components import (
    ExperimentalTransformer,
    ExperimentalTransformerKVCache,
    ExperimentalTransformerKVCacheDemo,
)


def _build_models(cls_kv, vocab_size: int = 32, pad_idx: int = 0):
    """Build a base transformer and a KV-cache variant with shared weights.

    Args:
        cls_kv: The KV-cache model class to instantiate.
        vocab_size: Output vocabulary size.
        pad_idx: Padding token index.

    Returns:
        tuple[ExperimentalTransformer, nn.Module]
            Base model and KV-cache variant with identical weights.
    """
    params = {
        "in_channels": 12,
        "vocab_size": vocab_size,
        "pad_idx": pad_idx,
        "d_model": 32,
        "nhead": 2,
        "num_encoder_layers": 1,
        "num_decoder_layers": 1,
        "dim_feedforward": 64,
        "dropout": 0.0,
    }

    base = ExperimentalTransformer(**params)
    kv = cls_kv(**params)

    # Share weights to ensure output parity is due to logic, not initialization
    kv.load_state_dict(base.state_dict(), strict=False)

    base.eval()
    kv.eval()
    return base, kv


@pytest.mark.parametrize(
    "kv_class", [ExperimentalTransformerKVCache, ExperimentalTransformerKVCacheDemo]
)
class TestKVCacheParity:
    """Test suite for verifying result parity between base and KV-cache models."""

    def test_forward_logits_match(self, kv_class) -> None:
        """Verify that a standard forward pass produces identical logits.

        Both models should behave identically when given the full target sequence.
        """
        torch.manual_seed(42)
        base, kv = _build_models(kv_class)

        B, T_src, T_tgt = 2, 16, 5
        src = torch.randn(B, T_src, 12)
        src_lengths = torch.tensor([T_src, T_src - 4], dtype=torch.long)
        tgt = torch.randint(low=1, high=32, size=(B, T_tgt))

        with torch.no_grad():
            out_base = base(src, src_lengths, tgt)
            out_kv = kv(src, src_lengths, tgt)

        assert torch.allclose(out_base, out_kv, atol=1e-6), (
            f"Logits mismatch for {kv_class.__name__}"
        )

    def test_generate_output_matches(self, kv_class) -> None:
        """Verify that autoregressive generation produces identical token sequences.

        Ensures that the incremental decoding logic used in generate() is correct.
        """
        torch.manual_seed(42)
        base, kv = _build_models(kv_class)

        B, T_src = 2, 16
        src = torch.randn(B, T_src, 12)
        src_lengths = torch.tensor([T_src, T_src - 4], dtype=torch.long)

        gen_kwargs = {"sos_idx": 1, "eos_idx": 2, "max_len": 10}

        with torch.no_grad():
            tokens_base = base.generate(src, src_lengths, **gen_kwargs)
            tokens_kv = kv.generate(src, src_lengths, **gen_kwargs)

        assert torch.equal(tokens_base, tokens_kv), (
            f"Generated tokens mismatch for {kv_class.__name__}"
        )


class TestExperimentalTransformerKVCache:
    """Detailed tests for the dictionary-based KV-cache implementation."""

    def test_decode_step_matches_full_decode(self) -> None:
        """Verify that incremental decode_step matches logits from a full decode pass.

        This test step-by-step compares incremental logits against the corresponding
        positions in the full-sequence decoder output.
        """
        torch.manual_seed(42)
        _, kv = _build_models(ExperimentalTransformerKVCache)

        B, T_src, T_tgt = 2, 16, 5
        src = torch.randn(B, T_src, 12)
        src_lengths = torch.tensor([T_src, T_src - 4], dtype=torch.long)
        tgt = torch.randint(low=1, high=32, size=(B, T_tgt))

        with torch.no_grad():
            memory, mem_mask = kv.encode(src, src_lengths)
            # 1. Full decode (standard transformer path)
            logits_full = kv.decode(tgt, memory, mem_mask)

            # 2. Incremental decode (KV-cache path)
            cache = kv.init_kv_cache(memory)
            logits_inc = []
            for t in range(T_tgt):
                token = tgt[:, t : t + 1]
                step_logits, cache = kv.decode_step(token, memory, mem_mask, cache)
                logits_inc.append(step_logits)

            logits_inc = torch.stack(logits_inc, dim=1)

        assert torch.allclose(logits_full, logits_inc, atol=1e-5), (
            "Incremental logits do not match full decode logits"
        )


class TestExperimentalTransformerKVCacheDemo:
    """Detailed tests for the tensor-stack-based KV-cache implementation (ONNX-friendly)."""

    def test_decode_step_matches_full_decode(self) -> None:
        """Verify that incremental decode_step matches logits from a full decode pass.

        Validates the specialized tensor-based caching logic used for ONNX export.
        """
        torch.manual_seed(42)
        _, kv = _build_models(ExperimentalTransformerKVCacheDemo)

        B, T_src, T_tgt = 2, 16, 5
        src = torch.randn(B, T_src, 12)
        src_lengths = torch.tensor([T_src, T_src - 4], dtype=torch.long)
        tgt = torch.randint(low=1, high=32, size=(B, T_tgt))

        with torch.no_grad():
            memory, mem_mask = kv.encode(src, src_lengths)
            # 1. Full decode (teacher-forced path)
            logits_full = kv.decode(tgt, memory, mem_mask)

            # 2. Incremental decode (KV-cache path)
            step, self_k, self_v, mem_k, mem_v = kv.init_kv_cache(memory)
            logits_inc = []
            for t in range(T_tgt):
                token = tgt[:, t : t + 1]
                step_logits, self_k, self_v = kv.decode_step(
                    token, step, self_k, self_v, mem_k, mem_v, mem_mask
                )
                logits_inc.append(step_logits)
                step += 1

            logits_inc = torch.stack(logits_inc, dim=1)

        assert torch.allclose(logits_full, logits_inc, atol=1e-5), (
            "Incremental logits do not match full decode logits"
        )
