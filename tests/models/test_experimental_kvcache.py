import importlib.util
from pathlib import Path

import torch
import torch.nn as nn


def _load_transformer_class(module_relative_path: str):
    root = Path(__file__).resolve().parents[2]
    module_path = root / module_relative_path
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ExperimentalTransformer


ExperimentalTransformerBase = _load_transformer_class(
    "src/hand_to_tex/models/components/experimental.py"
)
ExperimentalTransformerKV = _load_transformer_class(
    "src/hand_to_tex/models/components/experimental_kvcache.py"
)


def _build_models() -> tuple[nn.Module, nn.Module]:
    params = {
        "in_channels": 10,
        "vocab_size": 41,
        "pad_idx": 0,
        "d_model": 64,
        "nhead": 8,
        "num_encoder_layers": 2,
        "num_decoder_layers": 2,
        "dim_feedforward": 128,
        "dropout": 0.0,
    }

    base = ExperimentalTransformerBase(**params)
    kv = ExperimentalTransformerKV(**params)

    kv.load_state_dict(base.state_dict(), strict=True)

    base.eval()
    kv.eval()
    return base, kv


def _sample_inputs(batch_size: int = 2, src_len: int = 27, tgt_len: int = 8):
    torch.manual_seed(123)
    src = torch.randn(batch_size, src_len, 10)
    src_lengths = torch.tensor([src_len, src_len - 7], dtype=torch.long)

    # Avoid PAD token (0) to keep comparison focused on autoregressive path.
    tgt = torch.randint(1, 41, (batch_size, tgt_len), dtype=torch.long)
    return src, src_lengths, tgt


def test_forward_shape_and_values_match_base_model() -> None:
    base, kv = _build_models()
    src, src_lengths, tgt = _sample_inputs()

    out_base = base(src=src, src_lengths=src_lengths, tgt=tgt)
    out_kv = kv(src=src, src_lengths=src_lengths, tgt=tgt)

    assert out_base.shape == out_kv.shape == (src.size(0), tgt.size(1), 41)
    assert torch.allclose(out_base, out_kv, atol=1e-6, rtol=1e-5)


def test_decode_step_cache_shapes_grow_per_step() -> None:
    _base, kv = _build_models()
    src, src_lengths, _tgt = _sample_inputs(tgt_len=6)

    memory, mem_mask = kv.encode(src, src_lengths)
    cache = kv.init_kv_cache(memory)

    assert int(cache["step"]) == 0

    token = torch.ones((src.size(0), 1), dtype=torch.long)
    for step in range(4):
        logits, cache = kv.decode_step(
            tgt_last=token,
            memory=memory,
            memory_key_padding_mask=mem_mask,
            cache=cache,
        )

        assert logits.shape == (src.size(0), 41)
        assert int(cache["step"]) == step + 1

        for layer_cache in cache["layers"]:
            assert layer_cache["self_k"].shape[2] == step + 1
            assert layer_cache["self_v"].shape[2] == step + 1

        token = torch.argmax(logits, dim=-1, keepdim=True)


def test_decode_step_matches_full_decode_teacher_forced() -> None:
    _base, kv = _build_models()
    src, src_lengths, tgt = _sample_inputs(tgt_len=7)

    memory, mem_mask = kv.encode(src, src_lengths)

    full = kv.decode(tgt=tgt, memory=memory, memory_key_padding_mask=mem_mask)

    cache = kv.init_kv_cache(memory)
    step_logits = []

    for step in range(tgt.size(1)):
        last = tgt[:, step : step + 1]
        logits_last, cache = kv.decode_step(
            tgt_last=last,
            memory=memory,
            memory_key_padding_mask=mem_mask,
            cache=cache,
        )
        step_logits.append(logits_last.unsqueeze(1))

    incremental = torch.cat(step_logits, dim=1)

    assert incremental.shape == full.shape
    assert torch.allclose(incremental, full, atol=2e-4, rtol=1e-4)
