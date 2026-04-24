from pathlib import Path
from typing import Literal

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from lightning.pytorch.loggers import WandbLogger
from torch import Tensor
from torchmetrics import MeanMetric, MetricCollection
from torchmetrics.text import CharErrorRate, WordErrorRate

from hand_to_tex.models.components import ExperimentalTransformer
from hand_to_tex.utils import LatexVocab, logger


class HMELightningModule(pl.LightningModule):
    """Lightning module for handwriting-to-LaTeX transcription.

    This module wraps the transformer model, training loss, metric tracking,
    text generation, and checkpoint loading logic.
    """

    def __init__(
        self,
        vocab_path: str,
        pretrained_model_path: str | None = None,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_generate_len: int = 150,
        lr: float = 3e-4,
        label_smoothing: float = 0.1,
        weight_decay: float = 1e-4,
    ):
        """Initialize the model, vocabulary, metrics, and training settings.

        Parameters
        ----------
        vocab_path:
            Path to the vocabulary file used for token encoding and decoding.
        pretrained_model_path:
            Optional path to a checkpoint with pretrained weights.
        d_model:
            Transformer hidden size.
        nhead:
            Number of attention heads in the transformer.
        num_encoder_layers:
            Number of encoder layers.
        num_decoder_layers:
            Number of decoder layers.
        dim_feedforward:
            Size of the feedforward layer inside transformer blocks.
        dropout:
            Dropout probability used in the transformer.
        max_generate_len:
            Maximum length of generated token sequences.
        lr:
            Learning rate used by the optimizer and scheduler.
        label_smoothing:
            Label smoothing value used in cross-entropy loss.
        weight_decay:
            Weight decay applied to selected optimizer parameters.
        """
        super().__init__()

        self.vocab = LatexVocab.load(vocab_path)

        self.model = ExperimentalTransformer(
            in_channels=10,
            vocab_size=len(self.vocab),
            pad_idx=self.vocab.PAD,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.max_generate_len = max_generate_len
        self.lr = lr
        self.weight_decay = weight_decay
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.vocab.PAD,
            label_smoothing=label_smoothing,
        )
        if pretrained_model_path is not None:
            self._load_pretrained_model(pretrained_model_path=pretrained_model_path)

        metrics = MetricCollection(
            {
                "CER": CharErrorRate(),
                "TER": WordErrorRate(),
            }
        )
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")
        self.val_exact_match = MeanMetric()
        self.test_exact_match = MeanMetric()

        self.validation_samples = []
        self.test_samples = []

        self.save_hyperparameters(ignore=["model", "vocab"])

    def forward(self, src: Tensor, src_lengths: Tensor, tgt: Tensor):
        """Run a forward pass through the transformer.

        Parameters
        ----------
        src:
            Input feature tensor.
        src_lengths:
            Lengths of the input sequences.
        tgt:
            Decoder input tokens.

        Returns
        -------
        Tensor
            Decoder logits for each target position.
        """
        return self.model(src=src, src_lengths=src_lengths, tgt=tgt)

    @torch.inference_mode()
    def generate(self, src: Tensor, src_lengths: Tensor) -> Tensor:
        """Generate token sequences from input features.

        Parameters
        ----------
        src:
            Input feature tensor.
        src_lengths:
            Lengths of the input sequences.

        Returns
        -------
        Tensor
            Generated token ids for each item in the batch.
        """
        B = src.size(0)
        device = src.device

        memory, mem_mask = self.model.encode(src, src_lengths)  # type: ignore

        tgt = torch.full(
            (B, self.max_generate_len), fill_value=self.vocab.PAD, dtype=torch.long, device=device
        )
        tgt[:, 0] = self.vocab.SOS
        unfinished_seqs = torch.ones(B, dtype=torch.bool, device=device)

        for step in range(1, self.max_generate_len):
            current_tgt = tgt[:, :step]

            output = self.model.decode(
                tgt=current_tgt, memory=memory, memory_key_padding_mask=mem_mask
            )  # type: ignore

            next_token_probs = output[:, -1, :]
            next_token = torch.argmax(next_token_probs, dim=-1)

            tgt[:, step] = torch.where(unfinished_seqs, next_token, tgt[:, step])

            unfinished_seqs = unfinished_seqs & (next_token != self.vocab.EOS)

            if not unfinished_seqs.any():
                tgt = tgt[:, : step + 1]
                break

        return tgt

    def _shared_step(
        self, batch: tuple[Tensor, Tensor, Tensor, Tensor]
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute the model output and loss for a batch.

        Parameters
        ----------
        batch:
            Batch containing features, feature lengths, tokens, tokens lenghts.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Loss, model output, expected target tokens.
        """
        padded_features, feature_lengths, padded_tokens, _ = batch

        target_input = padded_tokens[:, :-1]
        target_expected = padded_tokens[:, 1:]

        output = self(src=padded_features, src_lengths=feature_lengths, tgt=target_input)

        loss = self.criterion(output.reshape(-1, output.shape[-1]), target_expected.reshape(-1))

        return loss, output, target_expected

    def _eval_step(
        self,
        batch: tuple[Tensor, Tensor, Tensor, Tensor],
        batch_idx: int,
        stage: Literal["val", "test"],
        metrics: MetricCollection,
        exact_match_metric: MeanMetric,
        sample_store: list[dict[str, str]],
    ) -> Tensor:
        """Run validation or test logic and collect example predictions.

        Parameters
        ----------
        batch:
            Batch returned by the validation or test dataloader.
        batch_idx:
            Index of the current batch.
        stage:
            Logging prefix
        metrics:
            Metric collection used to accumulate sequence scores.
        exact_match_metric:
            Exact-match metric aggregated across all batches in an epoch.
        sample_store:
            List used to store example predictions for the current epoch.

        Returns
        -------
        Tensor
            Computed loss for the batch.
        """
        loss, _, expected = self._shared_step(batch)

        padded_ft, ft_lengths, _, _ = batch
        generated_ts = self.generate(padded_ft, ft_lengths)

        predicted_txt = [self.vocab.tensor_to_str(ts) for ts in generated_ts]
        expected_txt = [self.vocab.tensor_to_str(ts) for ts in expected]

        metrics.update(predicted_txt, expected_txt)

        exact_hits = sum(
            int(pred == exp) for pred, exp in zip(predicted_txt, expected_txt, strict=True)
        )
        batch_size = len(predicted_txt)
        batch_exact_ratio = 100.0 * exact_hits / batch_size if batch_size > 0 else 0.0
        exact_match_metric.update(
            torch.tensor(batch_exact_ratio, dtype=torch.float32, device=self.device),
            weight=batch_size,
        )

        self.log(f"{stage}/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            f"{stage}/exact_match",
            exact_match_metric,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        if batch_idx == 0:
            for prediction, expected_value in zip(predicted_txt, expected_txt, strict=True):
                sample_store.append({"prediction": prediction, "expected": expected_value})

        return loss

    def training_step(self, batch, batch_idx):
        """Run one training step.

        Parameters
        ----------
        batch:
            Training batch returned by the dataloader.
        batch_idx:
            Index of the current batch.

        Returns
        -------
        Tensor
            Training loss.
        """
        loss, _, _ = self._shared_step(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Run one validation step and collect example predictions.

        Parameters
        ----------
        batch:
            Validation batch returned by the dataloader.
        batch_idx:
            Index of the current batch.

        Returns
        -------
        Tensor
            Validation loss.
        """
        return self._eval_step(
            batch=batch,
            batch_idx=batch_idx,
            stage="val",
            metrics=self.val_metrics,
            exact_match_metric=self.val_exact_match,
            sample_store=self.validation_samples,
        )

    def test_step(self, batch, batch_idx):
        """Run one test step and collect example predictions.

        Parameters
        ----------
        batch:
            Test batch returned by the dataloader.
        batch_idx:
            Index of the current batch.

        Returns
        -------
        Tensor
            Test loss.
        """
        return self._eval_step(
            batch=batch,
            batch_idx=batch_idx,
            stage="test",
            metrics=self.test_metrics,
            exact_match_metric=self.test_exact_match,
            sample_store=self.test_samples,
        )

    def on_validation_epoch_start(self) -> None:
        """Reset collected validation samples at the start of an epoch."""
        self.validation_samples = []

    def on_validation_epoch_end(self) -> None:
        """Log a small table of validation predictions to Weights & Biases."""
        if not self.trainer.sanity_checking:
            if self.logger and isinstance(self.logger, WandbLogger) and self.validation_samples:
                cols = ["Epoch", "Predicted", "Expected"]
                data = [
                    (self.current_epoch, s["prediction"], s["expected"])
                    for s in self.validation_samples
                ]
                table = wandb.Table(columns=cols, data=data)

                self.logger.experiment.log({"validation_prediction_table": table})
        self.validation_samples = []

    def on_test_epoch_start(self) -> None:
        """Reset collected test samples at the start of an epoch."""
        self.test_samples = []

    def on_test_epoch_end(self) -> None:
        """Log a small table of test predictions to Weights & Biases."""
        if not self.trainer.sanity_checking:
            if self.logger and isinstance(self.logger, WandbLogger) and self.test_samples:
                cols = ["Epoch", "Predicted", "Expected"]
                data = [
                    (self.current_epoch, s["prediction"], s["expected"]) for s in self.test_samples
                ]
                table = wandb.Table(columns=cols, data=data)

                self.logger.experiment.log({"test_prediction_table": table})
        self.test_samples = []

    def on_load_checkpoint(self, checkpoint: dict):
        """Clean checkpoint keys before Lightning restores the state.

        Parameters
        ----------
        checkpoint:
            Checkpoint dictionary passed by Lightning.
        """

        if (state_dict := checkpoint.get("state_dict")) is None:
            logger.warning("Couldn't find `state_dict` while loading the checkpoint")
        else:
            checkpoint["state_dict"] = HMELightningModule._clean_state_dict(state_dict=state_dict)

    def on_save_checkpoint(self, checkpoint: dict):
        """Before saving the checkpoint, clean the state dict from compilation
        changes such as _orig_mod in weights' names

        Parameters
        ----------
        checkpoint:
            Checkpoint dictionary passed by Lightning.
        """

        if (state_dict := checkpoint.get("state_dict")) is None:
            logger.warning("Couldn't find `state_dict` while saving the checkpoint")
        else:
            checkpoint["state_dict"] = HMELightningModule._clean_state_dict(state_dict=state_dict)

    def configure_optimizers(self):
        """Create the optimizer and learning-rate scheduler.

        Returns
        -------
        dict
            Optimizer configuration understood by Lightning.
        """
        decay = set()
        no_decay = set()

        whitelist_weight_modules = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.MultiheadAttention)
        blacklist_weight_modules = (nn.LayerNorm, nn.BatchNorm1d, nn.Embedding)

        for mn, m in self.model.named_modules():
            for pn, _p in m.named_parameters(recurse=False):
                fpn = f"{mn}.{pn}" if mn else pn

                if pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)
                elif pn.endswith("in_proj_weight"):
                    decay.add(fpn)

        param_dict = dict(self.model.named_parameters())
        inter_params = decay & no_decay
        assert len(inter_params) == 0, f"Colliding optimizer weights: {inter_params}"
        union_params = decay | no_decay
        for pn, _p in param_dict.items():
            if pn not in union_params:
                no_decay.add(pn)

        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(decay)],
                "weight_decay": self.hparams.weight_decay,  # type: ignore
            },
            {"params": [param_dict[pn] for pn in sorted(no_decay)], "weight_decay": 0.0},
        ]

        optimizer = optim.AdamW(optim_groups, lr=self.lr)

        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,
            total_steps=self.trainer.estimated_stepping_batches,  # type: ignore
            pct_start=0.1,
            anneal_strategy="cos",
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def _load_pretrained_model(self, pretrained_model_path: str, strict: bool = True):
        """Load pretrained weights from a checkpoint file.

        Parameters
        ----------
        pretrained_model_path:
            Path to a checkpoint file containing a state dict.
        strict:
            Whether to require an exact key match when loading weights.
        """
        try:
            pth = Path(pretrained_model_path)
            pretrained_model = torch.load(
                pth, map_location=lambda storage, loc: storage, weights_only=False
            )

            if (state_dict := pretrained_model.get("state_dict")) is None:
                logger.warning(f"Couldn't find `state_dict` in {pretrained_model_path}")
            else:
                state_dict = HMELightningModule._clean_state_dict(state_dict=state_dict)
                self.load_state_dict(state_dict=state_dict, strict=strict)
        except Exception as e:
            raise e

    @staticmethod
    def _clean_state_dict(state_dict: dict) -> dict:
        """Clean state_dict keys from compilation additions (_orig_mod)

        Parameters
        ----------
        state_dict:
            state dictionary obtained from checkpoint

        Returns
        -------
        state_dict with cleaned keys
        """
        return {k.replace("._orig_mod.", "."): v for k, v in state_dict.items()}
