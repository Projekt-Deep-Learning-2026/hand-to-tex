import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from lightning.pytorch.loggers import WandbLogger
from torch import Tensor
from torchmetrics import MetricCollection
from torchmetrics.text import CharErrorRate, WordErrorRate

from hand_to_tex.models.components import ExperimentalTransformer
from hand_to_tex.utils import LatexVocab


class HMELightningModule(pl.LightningModule):
    def __init__(
        self,
        vocab_path: str,
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

        metrics = MetricCollection(
            {
                "CER": CharErrorRate(),
                "TER": WordErrorRate(),
            }
        )
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

        self.save_hyperparameters(ignore=["model", "vocab"])

    def forward(self, src: Tensor, src_lengths: Tensor, tgt: Tensor):
        return self.model(src=src, src_lengths=src_lengths, tgt=tgt)

    def _shared_step(
        self, batch: tuple[Tensor, Tensor, Tensor, Tensor]
    ) -> tuple[Tensor, Tensor, Tensor]:
        padded_features, feature_lengths, padded_tokens, _ = batch

        target_input = padded_tokens[:, :-1]
        target_expected = padded_tokens[:, 1:]

        output = self(src=padded_features, src_lengths=feature_lengths, tgt=target_input)

        loss = self.criterion(output.reshape(-1, output.shape[-1]), target_expected.reshape(-1))

        return loss, output, target_expected

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._shared_step(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _, expected = self._shared_step(batch)

        padded_ft, ft_lengths, _, _ = batch
        generated_ts = self.generate(padded_ft, ft_lengths)

        predicted_txt = [self._to_expr(ts) for ts in generated_ts]
        expected_txt = [self._to_expr(ts) for ts in expected]

        self.val_metrics.update(predicted_txt, expected_txt)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True, prog_bar=True)

        if batch_idx == 0:
            for p, e in zip(predicted_txt, expected_txt, strict=True):
                self.validation_samples.append({"prediction": p, "expected": e})

        return loss

    @torch.inference_mode()
    def generate(self, src: Tensor, src_lengths: Tensor) -> Tensor:
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

    def configure_optimizers(self):
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

    def _to_expr(self, tokens: Tensor) -> str:
        token_ids = tokens.tolist()
        expr = []
        for t_id in token_ids:
            match t_id:
                case self.vocab.EOS:
                    break
                case self.vocab.PAD | self.vocab.SOS | self.vocab.UNK:
                    continue
                case _:
                    expr.append(self.vocab.decode(t_id))

        return " ".join(expr)

    def on_validation_epoch_start(self) -> None:
        self.validation_samples = []

    def on_validation_epoch_end(self) -> None:

        if self.logger and isinstance(self.logger, WandbLogger) and self.validation_samples:
            cols = ["Epoch", "Predicted", "Expected"]
            data = [
                (self.current_epoch, s["prediction"], s["expected"])
                for s in self.validation_samples
            ]
            table = wandb.Table(columns=cols, data=data)

            self.logger.experiment.log({"validation_prediction_table": table})
            self.val_samples = []
