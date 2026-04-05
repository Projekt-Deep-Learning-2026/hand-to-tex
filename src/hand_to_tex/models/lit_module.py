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
        lr: float = 3e-4,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.model = model
        self.vocab = vocab
        self.lr = lr
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

    def forward(self, src: Tensor, src_lengths: Tensor, tgt: Tensor):
        return self.model(src=src, src_lengths=src_lengths, tgt=tgt)

    def _shared_step(
        self, batch: tuple[Tensor, Tensor, Tensor, Tensor]
    ) -> tuple[Tensor, Tensor, Tensor]:
        padded_features, feature_lenghts, padded_tokens, _ = batch

        target_input = padded_tokens[:, :-1]
        target_expected = padded_tokens[:, 1:]

        output = self(src=padded_features, src_lengths=feature_lenghts, tgt=target_input)

        loss = self.criterion(output.reshape(-1, output.shape[-1]), target_expected.reshape(-1))

        return loss, output, target_expected

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._shared_step(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, output, expected = self._shared_step(batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        predicted_ids = torch.argmax(output, dim=-1)

        predicted_text = [" ".join(self.vocab.decode_sequence(p.tolist())) for p in predicted_ids]
        expected_text = [" ".join(self.vocab.decode_sequence(e.tolist())) for e in expected]

        self.val_metrics.update(predicted_text, expected_text)

        return loss

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.lr)
