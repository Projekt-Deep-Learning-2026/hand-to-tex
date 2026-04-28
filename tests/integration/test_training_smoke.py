from __future__ import annotations

from pathlib import Path

import torch

from hand_to_tex.datasets.datamodule import HMELightningDataModule
from hand_to_tex.models.lit_module import HMELightningModule


def _build_trainer(default_root_dir: Path):
    import lightning.pytorch as pl

    return pl.Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=str(default_root_dir),
    )


class TestTrainingSmoke:
    def test_trainer_fit_runs_one_step_without_error(
        self,
        tmp_path: Path,
        tiny_model_kwargs: dict,
        preprocessed_pt_root: Path,
    ):
        torch.manual_seed(0)
        model = HMELightningModule(**tiny_model_kwargs)
        dm = HMELightningDataModule(
            root=str(preprocessed_pt_root),
            processed=True,
            batch_size=2,
            num_workers=0,
            pin_memory=False,
        )

        trainer = _build_trainer(tmp_path)
        trainer.fit(model, datamodule=dm)

        assert next(model.parameters()).device.type == "cpu"

    def test_trainer_test_runs_one_step_without_error(
        self,
        tmp_path: Path,
        tiny_model_kwargs: dict,
        preprocessed_pt_root: Path,
    ):
        torch.manual_seed(0)
        model = HMELightningModule(**tiny_model_kwargs)
        dm = HMELightningDataModule(
            root=str(preprocessed_pt_root),
            processed=True,
            batch_size=2,
            num_workers=0,
            pin_memory=False,
        )

        trainer = _build_trainer(tmp_path)
        results = trainer.test(model, datamodule=dm)

        assert isinstance(results, list)


class TestTrainingStepGradientFlow:
    def test_manual_train_loop_reduces_loss_over_a_few_steps(
        self,
        tiny_model_kwargs: dict,
        synthetic_batch,
    ):
        torch.manual_seed(0)
        model = HMELightningModule(**tiny_model_kwargs).train()
        optim = torch.optim.SGD(model.parameters(), lr=0.5)

        loss_before, _, _ = model._shared_step(synthetic_batch)
        for _ in range(5):
            loss, _, _ = model._shared_step(synthetic_batch)
            optim.zero_grad()
            loss.backward()
            optim.step()

        with torch.no_grad():
            loss_after, _, _ = model._shared_step(synthetic_batch)

        assert loss_after.item() < loss_before.item(), (
            f"loss did not decrease: {loss_before.item()} -> {loss_after.item()}"
        )
