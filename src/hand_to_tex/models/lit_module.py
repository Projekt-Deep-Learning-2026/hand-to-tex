import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torchmetrics import MetricCollection
from torchmetrics.text import CharErrorRate, WordErrorRate

from hand_to_tex.utils import LatexVocab


class HMELightningModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        vocab: LatexVocab,
        max_generate_len: int = 150,
        lr: float = 3e-4,
        label_smoothing: float = 0.1,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.model = model
        self.vocab = vocab
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

        return loss

    @torch.no_grad()
    def generate(self, src: Tensor, src_lengths: Tensor) -> Tensor:
        B = src.size(0)
        device = src.device

        tgt = torch.full((B, 1), fill_value=self.vocab.SOS, dtype=torch.long, device=device)
        for _ in range(self.max_generate_len):
            output = self.model(src=src, src_lengths=src_lengths, tgt=tgt)

            next_token_probs = output[:, -1, :]
            next_token_list = torch.argmax(next_token_probs, dim=-1).unsqueeze(1)

            tgt = torch.cat([tgt, next_token_list], dim=1)

            # check if all `columns` of tgt has End-of-sequence, break if so
            if (tgt == self.vocab.EOS).any(dim=1).all():
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
