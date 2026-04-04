import lightning as pl
import torch.nn as nn
import torch.optim as optim
from torch import Tensor

from hand_to_tex.utils import LatexVocab


class HMELightningModule(pl.LightningModule):
    def __init__(self, model, vocab: LatexVocab, lr: float = 3e-4, label_smoothing: float = 0.1):
        super().__init__()
        self.model = model
        self.vocab = vocab
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.vocab.PAD,
            label_smoothing=label_smoothing,
        )

    def forward(self, src, src_lengths, tgt):
        return self.model(src=src, src_lengths=src_lengths, tgt=tgt)

    def _shared_step(self, batch: tuple[Tensor, Tensor, Tensor, Tensor]):
        padded_ft, ft_lengths, padded_ts, _ = batch
        tgt_input = padded_ts[:, :-1]
        tgt_expected = padded_ts[:, 1:]
        output = self(src=padded_ft, src_lengths=ft_lengths, tgt=tgt_input)
        loss = self.criterion(output.reshape(-1, output.shape[-1]), tgt_expected.reshape(-1))
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.lr)
