import os
from importlib import import_module

import torch
from lightning.pytorch.cli import LightningCLI

from hand_to_tex.datasets.datamodule import HMELightningDataModule
from hand_to_tex.models.components.base import BaseDecoderModel
from hand_to_tex.models.lit_module import HMELightningModule
from hand_to_tex.utils import LatexVocab, logger


class HandToTexCLI(LightningCLI):
    def before_instantiate_classes(self):

        torch.set_float32_matmul_precision("high")
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        cfg = getattr(self, "config", {})

        if not (model_cfg := cfg.get("model")):
            raise ValueError("Expected `model` section in config")
        elif not isinstance(model_cfg, dict):
            raise ValueError(
                f"Expected `model` section of config to be a dict, received {type(cfg['model'])}"
            )

        model_instance = self._create_model_instance(config=model_cfg)

        self.instantiate_kwargs = getattr(self, "instantiate_kwargs", {})
        self.instantiate_kwargs["model"] = model_instance

    def before_fit(self):
        try:
            logger.info("Starting torch.compile...")
            self.model.model = torch.compile(self.model.model)  # type: ignore
        except Exception as e:
            logger.warning(f"Error during compilation: {e}, continuing on uncompiled model")
            pass

    @staticmethod
    def _create_model_instance(config: dict) -> BaseDecoderModel:

        if not (class_path := config.get("class_path")):
            raise ValueError("Expected model configuration to have a `class_path` parameter")
        elif not (model_args := config.get("init_args")):
            raise ValueError("Expected model configuration to have a `init_class` section")
        elif not isinstance(model_args, dict):
            raise ValueError(
                f"Expected model.init_args to be a `dict` instance, received {type(model_args)}"
            )
        if not (vocab_path := config.get("vocab_path")):
            raise ValueError("Expected model configuration to have a `vocab_path` parameter")

        vocab = LatexVocab.load(path=vocab_path)
        model_args["vocab_size"] = len(vocab)
        model_args["pad_idx"] = vocab.PAD

        module_name, class_name = class_path.rsplit(".", 1)
        mod = import_module(module_name)
        ModelClass = getattr(mod, class_name)
        model_instance: BaseDecoderModel = ModelClass(**model_args)

        return model_instance


def main():
    HandToTexCLI(
        model_class=HMELightningModule,
        datamodule_class=HMELightningDataModule,
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":
    main()
