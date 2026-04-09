import os

import torch
from lightning.pytorch.cli import LightningCLI

from src.hand_to_tex.datasets.datamodule import HMELightningDataModule
from src.hand_to_tex.models.lit_module import HMELightningModule


class HandToTexCLI(LightningCLI):
    def before_instantiate_classes(self):
        torch.set_float32_matmul_precision("high")

        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    def before_fit(self):
        self.model.model = torch.compile(self.model.model, mode="max-autotune")  # type: ignore


def main():

    HandToTexCLI(
        model_class=HMELightningModule,
        datamodule_class=HMELightningDataModule,
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":
    main()
