"""Tests for ONNX export and inference parity.

Verifies that models can be exported to ONNX and produce identical results
in ONNX Runtime as they do in PyTorch. Covers dynamic axes and different
model architectures.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch

from hand_to_tex.models.components import (
    ExperimentalTransformer,
    ExperimentalTransformerKVCacheDemo,
)
from hand_to_tex.models.components.exportable import OnnxExportable
from hand_to_tex.models.lit_module import HMELightningModule
from hand_to_tex.utils import LatexVocab

ModelArchitectures = ExperimentalTransformer | ExperimentalTransformerKVCacheDemo


def _build_lit_module(model_class: type[ModelArchitectures], vocab_path: str) -> HMELightningModule:
    """Build a tiny HMELightningModule for testing.

    Args:
        model_class: The class of the model component to instantiate.
        vocab_path: Path to the vocabulary file.

    Returns:
        HMELightningModule with a tiny version of the requested model.
    """
    torch.manual_seed(42)
    # Ensure the model class can be instantiated with standard params
    model = model_class(
        in_channels=12,
        vocab_size=32,
        pad_idx=0,
        d_model=32,
        nhead=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=64,
        dropout=0.0,
    )
    # Cast to nn.Module for HMELightningModule compatibility if needed,
    # but here we know these models are nn.Modules.
    assert isinstance(model, torch.nn.Module)
    return HMELightningModule(
        vocab_path=vocab_path,
        model=model,
        max_generate_len=8,
    )


@pytest.mark.parametrize(
    "model_class",
    [
        pytest.param(
            ExperimentalTransformer,
            marks=pytest.mark.xfail(
                reason="Standard nn.Transformer export is brittle with dynamic shapes"
            ),
        ),
        ExperimentalTransformerKVCacheDemo,
    ],
)
class TestOnnxExport:
    """Test suite for ONNX export functionality and runtime parity."""

    def test_export_to_onnx_creates_files(
        self, model_class: type[ModelArchitectures], vocab_path: str, tmp_path: Path
    ) -> None:
        """Verify that export_to_onnx generates the expected .onnx files.

        The model must implement the OnnxExportable contract.
        """
        module = _build_lit_module(model_class, vocab_path)
        out_dir = tmp_path / "onnx_test"

        created_paths = module.export_to_onnx(out_dir=out_dir)

        assert len(created_paths) > 0
        for name, path in created_paths.items():
            assert path.exists(), f"Exported file {path} does not exist"
            assert path.suffix == ".onnx", f"Exported file {path} has wrong extension"
            assert name in path.name, (
                f"Exported file {path} name does not contain module name '{name}'"
            )

    def test_onnx_parity_inference(
        self,
        model_class: type[ModelArchitectures],
        vocab: LatexVocab,
        vocab_path: str,
        tmp_path: Path,
    ) -> None:
        """Verify that ONNX Runtime inference matches PyTorch inference results exactly.

        Ensures that the exported model preserves the behavioral correctness of the original.
        """
        torch.manual_seed(42)
        module = _build_lit_module(model_class, vocab_path).eval()
        out_dir = tmp_path / "onnx_parity"

        # Export model to ONNX
        created_paths = module.export_to_onnx(out_dir=out_dir)

        # Initialize ONNX Runtime sessions
        sessions = {
            name: ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            for name, path in created_paths.items()
        }

        # Prepare test inputs (Batch Size = 1)
        B, T_src = 1, 16
        src = torch.randn(B, T_src, 12)
        src_lengths = torch.tensor([T_src], dtype=torch.long)

        # 1. Run PyTorch inference
        with torch.no_grad():
            expected_tokens = module.model.generate(
                src,
                src_lengths,
                sos_idx=vocab.SOS,
                eos_idx=vocab.EOS,
                max_len=8,
            )
            expected_results = [vocab.decode_sequence(seq.tolist()) for seq in expected_tokens]

        # 2. Run ONNX Runtime inference
        # Explicit check for mypy/runtime safety
        assert isinstance(module.model, OnnxExportable)
        actual_results = model_class.run_onnx_inference(
            sessions=sessions,
            src_features=src,
            src_lengths=src_lengths,
            vocab=vocab,
            max_len=8,
        )

        # 3. Assert parity
        assert actual_results == expected_results, (
            "ONNX inference results differ from PyTorch results"
        )


class TestOnnxDynamicAxes:
    """Test suite for verifying dynamic axis support in exported ONNX models."""

    def test_dynamic_batch_size(self, vocab_path: str, tmp_path: Path) -> None:
        """Verify that the exported encoder handles varying batch sizes.

        Tests dynamic axis 'batch' for the encoder input 'src'.
        """
        module = _build_lit_module(ExperimentalTransformerKVCacheDemo, vocab_path).eval()
        out_dir = tmp_path / "onnx_dynamic"
        created_paths = module.export_to_onnx(out_dir=out_dir)

        enc_sess = ort.InferenceSession(
            str(created_paths["encoder"]), providers=["CPUExecutionProvider"]
        )

        for batch_size in [1, 2, 4]:
            # Use different sequence lengths too to verify T axis
            T = 20 + batch_size
            src = np.random.randn(batch_size, T, 12).astype(np.float32)
            src_lengths = np.array([T] * batch_size, dtype=np.int64)

            # ort.InferenceSession.run returns list[numpy.ndarray]
            outputs = enc_sess.run(None, {"src": src, "src_lengths": src_lengths})

            if not isinstance(outputs, list) or len(outputs) == 0:
                pytest.fail("ONNX session run did not return expected list of outputs")

            memory = outputs[0]
            assert isinstance(memory, np.ndarray), "First output must be a numpy array"
            assert memory.shape[0] == batch_size, (
                f"Expected batch size {batch_size}, got {memory.shape[0]}"
            )
