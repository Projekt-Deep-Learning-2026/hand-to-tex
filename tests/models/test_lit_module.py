"""Tests for HMELightningModule.

Validates initialization, forward pass, loss computation, and generation.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from hand_to_tex.models.lit_module import HMELightningModule


class TestModuleInitialization:
    """Test module setup and configuration."""

    def test_module_components_exist(self, tiny_lit_module: HMELightningModule) -> None:
        """Module must have core components."""
        assert hasattr(tiny_lit_module, "model")
        assert hasattr(tiny_lit_module, "vocab")
        assert isinstance(tiny_lit_module.criterion, nn.CrossEntropyLoss)

    def test_criterion_configuration(self, tiny_lit_module: HMELightningModule) -> None:
        """Loss function must ignore PAD tokens."""
        assert tiny_lit_module.criterion.ignore_index == tiny_lit_module.vocab.PAD


class TestForward:
    """Test the forward pass."""

    def test_forward_shape(
        self, tiny_lit_module: HMELightningModule, synthetic_batch: tuple
    ) -> None:
        """Forward must return logits of shape (B, T-1, V)."""
        tiny_lit_module.eval()
        features, lengths, tokens, _ = synthetic_batch
        target = tokens[:, :-1]

        out = tiny_lit_module(src=features, src_lengths=lengths, tgt=target)

        B, T = target.shape
        V = len(tiny_lit_module.vocab)
        assert out.shape == (B, T, V)
        assert out.dtype == torch.float32

    def test_forward_output_finite(
        self, tiny_lit_module: HMELightningModule, synthetic_batch: tuple
    ) -> None:
        """Output must not contain NaN or Inf."""
        tiny_lit_module.eval()
        features, lengths, tokens, _ = synthetic_batch

        out = tiny_lit_module(src=features, src_lengths=lengths, tgt=tokens[:, :-1])

        assert torch.isfinite(out).all()


class TestSharedStep:
    """Test the _shared_step method used by training/validation/test."""

    def test_shared_step_returns_correct_types(
        self, tiny_lit_module: HMELightningModule, synthetic_batch: tuple
    ) -> None:
        """_shared_step must return (loss, output, target)."""
        tiny_lit_module.train()
        loss, output, target = tiny_lit_module._shared_step(synthetic_batch)

        assert loss.ndim == 0  # Scalar
        assert torch.isfinite(loss)
        assert loss >= 0.0

        B, T_tgt = synthetic_batch[2].shape
        V = len(tiny_lit_module.vocab)

        assert output.shape == (B, T_tgt - 1, V)
        assert target.shape == (B, T_tgt - 1)

    def test_perfect_logits_give_zero_loss(
        self, tiny_lit_module: HMELightningModule, synthetic_batch: tuple
    ) -> None:
        """When logits perfectly match targets, loss should be ~zero."""
        _, _, tokens, _ = synthetic_batch
        target = tokens[:, 1:]
        B, T = target.shape
        V = len(tiny_lit_module.vocab)

        # Create perfect logits: very high value for correct class
        perfect_logits = torch.full((B, T, V), -1e9)
        perfect_logits.scatter_(dim=2, index=target.unsqueeze(-1), value=0.0)

        loss = tiny_lit_module.criterion(
            perfect_logits.reshape(-1, V),
            target.reshape(-1),
        )
        assert loss.item() < 1e-3

    def test_pad_tokens_ignored_in_loss(self, tiny_lit_module: HMELightningModule) -> None:
        """Loss at PAD positions must be ignored."""
        vocab = tiny_lit_module.vocab
        B, T, V = 2, 3, len(vocab)

        target = torch.tensor(
            [
                [vocab.encode("x"), vocab.EOS, vocab.encode("x")],
                [vocab.encode("x"), vocab.EOS, vocab.PAD],
            ],
            dtype=torch.long,
        )

        logits = torch.randn(B, T, V, requires_grad=False)
        loss_before = tiny_lit_module.criterion(
            logits.reshape(-1, V),
            target.reshape(-1),
        )

        # Perturb PAD position
        logits_modified = logits.clone()
        logits_modified[1, 2, :] = torch.randn(V) * 1000
        loss_after = tiny_lit_module.criterion(
            logits_modified.reshape(-1, V),
            target.reshape(-1),
        )

        # Losses should be the same
        assert torch.isclose(loss_before, loss_after, atol=1e-5)

    def test_loss_decreases_after_step(
        self, tiny_lit_module: HMELightningModule, synthetic_batch: tuple
    ) -> None:
        """After one optimizer step, loss should decrease."""
        tiny_lit_module.train()
        optim = torch.optim.SGD(tiny_lit_module.parameters(), lr=0.1)

        loss_before, _, _ = tiny_lit_module._shared_step(synthetic_batch)
        optim.zero_grad()
        loss_before.backward()
        optim.step()

        with torch.no_grad():
            loss_after, _, _ = tiny_lit_module._shared_step(synthetic_batch)

        assert loss_after < loss_before

    def test_gradients_flow_to_all_params(
        self, tiny_lit_module: HMELightningModule, synthetic_batch: tuple
    ) -> None:
        """All parameters must receive gradients."""
        tiny_lit_module.train()
        loss, _, _ = tiny_lit_module._shared_step(synthetic_batch)
        loss.backward()

        missing = [
            name
            for name, p in tiny_lit_module.named_parameters()
            if p.requires_grad and p.grad is None
        ]
        assert not missing, f"Missing gradients: {missing}"


class TestTrainingStep:
    """Test the training_step method."""

    def test_training_step_returns_scalar_loss(
        self, tiny_lit_module: HMELightningModule, synthetic_batch: tuple
    ) -> None:
        """training_step must return scalar tensor."""
        tiny_lit_module.train()
        tiny_lit_module.log = lambda *a, **kw: None  # Mock logging

        loss = tiny_lit_module.training_step(synthetic_batch, batch_idx=0)

        assert loss.ndim == 0
        assert loss.requires_grad
        assert torch.isfinite(loss)


class TestGeneration:
    """Test sequence generation."""

    def test_generate_returns_tokens(
        self, tiny_lit_module: HMELightningModule, synthetic_batch: tuple
    ) -> None:
        """generate() must return token tensor."""
        tiny_lit_module.eval()
        features, lengths, *_ = synthetic_batch

        tokens = tiny_lit_module.generate(features, lengths)

        assert tokens.dtype == torch.long
        assert tokens.ndim == 2
        assert tokens.shape[0] == features.shape[0]
        assert tokens.shape[1] <= tiny_lit_module.max_generate_len

    def test_generate_starts_with_sos(
        self, tiny_lit_module: HMELightningModule, synthetic_batch: tuple
    ) -> None:
        """Generated sequences must start with SOS token."""
        tiny_lit_module.eval()
        features, lengths, *_ = synthetic_batch

        tokens = tiny_lit_module.generate(features, lengths)

        assert (tokens[:, 0] == tiny_lit_module.vocab.SOS).all()

    def test_generate_tokens_in_vocab(
        self, tiny_lit_module: HMELightningModule, synthetic_batch: tuple
    ) -> None:
        """Generated tokens must be valid."""
        tiny_lit_module.eval()
        features, lengths, *_ = synthetic_batch

        tokens = tiny_lit_module.generate(features, lengths)
        V = len(tiny_lit_module.vocab)

        assert (tokens >= 0).all() and (tokens < V).all()
