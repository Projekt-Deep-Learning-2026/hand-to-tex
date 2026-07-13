import torch
from lightning.pytorch.cli import LightningCLI

from tex_gen.datasets.datamodule import TexGenDataModule

# Importy klas z Twojej paczki tex_gen
from tex_gen.models import TexGenLightningModule


def cli_main():
    _cli = LightningCLI(
        model_class=TexGenLightningModule,
        datamodule_class=TexGenDataModule,
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    cli_main()
