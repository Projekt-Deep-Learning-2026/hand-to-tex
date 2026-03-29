import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from hand_to_tex.utils import LatexVocab


class BaselineTrainer:
    """
    Controller architecture for conducting supervised training loops over the BaselineTransformer.

    This overarching class handles the initialization and orchestration of network optimization
    mechanisms, applying cross-entropy optimization while managing dynamic data loads, validation
    steps, and hardware integrations.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        vocab: LatexVocab,
        device: torch.device,
        lr: float = 3e-4,
    ):
        """
        Initializes the optimization environments for the training instance.

        Parameters
        ----------
        model : nn.Module
            The neural network architecture designated for parameter adjustments via backpropagation.
        train_loader : DataLoader
            The iterative data pipeline supplying batches of features and labels for the training set.
        valid_loader : DataLoader
            The iterative data pipeline supplying batches of features and labels for the validation set.
        vocab : LatexVocab
            The initialized dictionary resolving conversions between discrete language tokens and tensors.
        device : torch.device
            The allocated hardware component dictating isolated tensor math computation targeting.
        lr : float, optional
            The base learning rate configured to scale adaptive moment optimization steps.
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.vocab = vocab
        self.device = device

        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)

        self.criterion = nn.CrossEntropyLoss(ignore_index=self.vocab.PAD, label_smoothing=0.1)

    def train_epoch(self) -> float:
        """
        Executes a singular comprehensive pass through the complete provisioned training dataset.

        Each batch experiences feature allocations, localized teacher forcing offsets, raw computational
        evaluations, error calculations through cross-entropy comparisons, and downstream network updates
        via backpropagated gradient scaling. Gradient normative clipping is natively enforced to preempt
        mathematical explosions common within auto-regressive contexts.

        Returns
        -------
        float
            The mathematically derived average loss magnitude spanning the entire isolated sequence.
        """
        self.model.train()
        total_loss = 0.0

        for batch_idx, batch in enumerate(self.train_loader):
            padded_ft, ft_lengths, padded_ts, ts_lengths = batch

            padded_ft = padded_ft.to(self.device)
            padded_ts = padded_ts.to(self.device)

            tgt_input = padded_ts[:, :-1]
            tgt_expected = padded_ts[:, 1:]

            self.optimizer.zero_grad()

            output = self.model(src=padded_ft, src_lengths=ft_lengths, tgt=tgt_input)

            output_flat = output.reshape(-1, output.shape[-1])
            tgt_flat = tgt_expected.reshape(-1)

            loss = self.criterion(output_flat, tgt_flat)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()

            if (batch_idx + 1) % 10 == 0:
                print(
                    f"   [Batch {batch_idx + 1}/{len(self.train_loader)}] Loss: {loss.item():.4f}"
                )

        if len(self.train_loader) == 0:
            return 0
        return total_loss / len(self.train_loader)

    def validate(self) -> float:
        """
        Executes an isolated, gradient-detached pass across the reserved validation sequences.

        The network assumes an operational evaluation mode, locking active dropout masks and batch
        normalizations, whilst bypassing autograd graph creation to maximize speed and minimalize
        memory overheads during objective performance appraisals.

        Returns
        -------
        float
            The mathematically derived average loss magnitude spanning the entire isolated sequence.
        """
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in self.valid_loader:
                padded_ft, ft_lengths, padded_ts, ts_lengths = batch

                padded_ft = padded_ft.to(self.device)
                padded_ts = padded_ts.to(self.device)

                tgt_input = padded_ts[:, :-1]
                tgt_expected = padded_ts[:, 1:]

                output = self.model(src=padded_ft, src_lengths=ft_lengths, tgt=tgt_input)

                output_flat = output.reshape(-1, output.shape[-1])
                tgt_flat = tgt_expected.reshape(-1)

                loss = self.criterion(output_flat, tgt_flat)
                total_loss += loss.item()

        if len(self.valid_loader) == 0:
            return 0
        return total_loss / len(self.valid_loader)

    def train(self, num_epochs: int):
        """
        Triggers and sustains the macroscopic iteration spanning multiple cascading epochs.

        Continuously invokes synchronous training and validation routines while dynamically logging
        intermittent hardware execution latencies and granular model trajectory statistics.

        Parameters
        ----------
        num_epochs : int
            The absolute limit denoting integer passes through the entire structural dataset array.
        """
        print(f"========== ROZPOCZĘTO SZKOLENIE ({num_epochs} epok) ==========")

        for epoch in range(num_epochs):
            start_time = time.time()

            print(f"\n[ EPOKA {epoch + 1}/{num_epochs} ]")
            train_loss = self.train_epoch()
            valid_loss = self.validate()

            epoch_time = time.time() - start_time
            print(f"🎯 Statystyki Epoki: Zakończono po {epoch_time:.2f}s")
            print(
                f"📉 Błąd Trenowania (Train Loss): {train_loss:.4f}  |  👀 Błąd Walidacji (Valid Loss): {valid_loss:.4f}"
            )
