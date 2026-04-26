from __future__ import annotations

import torch
import torch.nn as nn

from hand_to_tex.models.lit_module import HMELightningModule


class TestLitModuleConstruction:
    def test_module_has_expected_components(self, tiny_lit_module: HMELightningModule):
        assert hasattr(tiny_lit_module, "model")
        assert hasattr(tiny_lit_module, "vocab")
        assert isinstance(tiny_lit_module.criterion, nn.CrossEntropyLoss)

    def test_criterion_ignores_pad_index(self, tiny_lit_module: HMELightningModule):
        """PAD tokens must not contribute to the gradient."""
        assert tiny_lit_module.criterion.ignore_index == tiny_lit_module.vocab.PAD


class TestForwardThroughLitModule:
    def test_forward_returns_logits(self, tiny_lit_module: HMELightningModule, synthetic_batch):
        tiny_lit_module.eval()
        padded_features, feature_lengths, padded_tokens, _ = synthetic_batch
        target_input = padded_tokens[:, :-1]

        out = tiny_lit_module(
            src=padded_features,
            src_lengths=feature_lengths,
            tgt=target_input,
        )

        B, T_tgt_minus_1 = target_input.shape
        assert out.shape == (B, T_tgt_minus_1, len(tiny_lit_module.vocab))


class TestLossComputation:
    def test_shared_step_returns_finite_scalar_loss(
        self, tiny_lit_module: HMELightningModule, synthetic_batch
    ):
        tiny_lit_module.train()
        loss, output, expected = tiny_lit_module._shared_step(synthetic_batch)

        assert loss.ndim == 0
        assert torch.isfinite(loss)
        assert loss.item() >= 0.0
        B, T_tgt = synthetic_batch[2].shape
        assert output.shape == (B, T_tgt - 1, len(tiny_lit_module.vocab))
        assert expected.shape == (B, T_tgt - 1)

    def test_loss_is_zero_when_logits_perfectly_predict_targets(
        self, tiny_lit_module: HMELightningModule, synthetic_batch
    ):
        _, _, padded_tokens, _ = synthetic_batch
        target_expected = padded_tokens[:, 1:]
        vocab_size = len(tiny_lit_module.vocab)

        B, T = target_expected.shape
        perfect_logits = torch.full((B, T, vocab_size), -1e9)
        perfect_logits.scatter_(dim=2, index=target_expected.unsqueeze(-1), value=0.0)

        loss = tiny_lit_module.criterion(
            perfect_logits.reshape(-1, vocab_size),
            target_expected.reshape(-1),
        )
        assert loss.item() < 1e-3

    def test_loss_decreases_after_one_optimizer_step(
        self, tiny_lit_module: HMELightningModule, synthetic_batch
    ):
        tiny_lit_module.train()
        # Plain SGD avoids needing configure_optimizers (which needs self.trainer).
        optim = torch.optim.SGD(tiny_lit_module.parameters(), lr=0.1)

        loss_before, _, _ = tiny_lit_module._shared_step(synthetic_batch)
        optim.zero_grad()
        loss_before.backward()
        optim.step()

        with torch.no_grad():
            loss_after, _, _ = tiny_lit_module._shared_step(synthetic_batch)

        assert loss_after.item() < loss_before.item()

    def test_pad_position_logits_do_not_affect_loss(self, tiny_lit_module: HMELightningModule):
        """Logits at PAD positions must be ignored by the criterion."""
        torch.manual_seed(0)
        vocab = tiny_lit_module.vocab
        eos, pad = vocab.EOS, vocab.PAD
        a = vocab.encode("x")
        V = len(vocab)

        target_expected = torch.tensor(
            [[a, eos, a], [a, eos, pad]],
            dtype=torch.long,
        )
        logits = torch.randn(2, 3, V, requires_grad=False)
        loss_before = tiny_lit_module.criterion(logits.reshape(-1, V), target_expected.reshape(-1))

        logits_perturbed = logits.clone()
        logits_perturbed[1, 2, :] = torch.randn(V) * 1000
        loss_after = tiny_lit_module.criterion(
            logits_perturbed.reshape(-1, V), target_expected.reshape(-1)
        )

        assert torch.isclose(loss_before, loss_after), (
            "Loss changed when only PAD-position logits were perturbed; "
            "ignore_index=PAD is likely not configured correctly."
        )


class TestTrainingStep:
    def test_training_step_returns_scalar_loss(
        self, tiny_lit_module: HMELightningModule, synthetic_batch
    ):
        tiny_lit_module.train()
        # Disable .log to avoid the "Trainer not attached" error.
        tiny_lit_module.log = lambda *_a, **_kw: None  # type: ignore[assignment]

        loss = tiny_lit_module.training_step(synthetic_batch, batch_idx=0)
        assert loss.ndim == 0
        assert loss.requires_grad
        assert torch.isfinite(loss)


class TestGenerate:
    def test_generate_returns_long_token_tensor(
        self, tiny_lit_module: HMELightningModule, synthetic_batch
    ):
        tiny_lit_module.eval()
        padded_features, feature_lengths, *_ = synthetic_batch

        tokens = tiny_lit_module.generate(padded_features, feature_lengths)

        assert tokens.dtype == torch.long
        assert tokens.ndim == 2
        assert tokens.shape[0] == padded_features.shape[0]
        assert tokens.shape[1] <= tiny_lit_module.max_generate_len

    def test_generate_starts_each_sequence_with_sos(
        self, tiny_lit_module: HMELightningModule, synthetic_batch
    ):
        tiny_lit_module.eval()
        padded_features, feature_lengths, *_ = synthetic_batch

        tokens = tiny_lit_module.generate(padded_features, feature_lengths)
        assert torch.all(tokens[:, 0] == tiny_lit_module.vocab.SOS)


class TestToExpr:
    def test_to_expr_stops_at_eos(self, tiny_lit_module: HMELightningModule):
        vocab = tiny_lit_module.vocab
        tokens = torch.tensor(
            [vocab.SOS, vocab.encode("x"), vocab.EOS, vocab.encode("y")],
            dtype=torch.long,
        )
        result = tiny_lit_module._to_expr(tokens)
        assert "y" not in result
        assert "x" in result

    def test_to_expr_skips_special_tokens(self, tiny_lit_module: HMELightningModule):
        vocab = tiny_lit_module.vocab
        tokens = torch.tensor(
            [vocab.SOS, vocab.PAD, vocab.UNK, vocab.encode("x"), vocab.EOS],
            dtype=torch.long,
        )
        result = tiny_lit_module._to_expr(tokens)
        assert "<SOS>" not in result
        assert "<PAD>" not in result
        assert "<UNK>" not in result
