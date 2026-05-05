import os
from importlib import import_module

import torch
from lightning.pytorch.cli import LightningCLI

from hand_to_tex.datasets.datamodule import HMELightningDataModule
from hand_to_tex.models.lit_module import HMELightningModule
from hand_to_tex.utils import LatexVocab


class HandToTexCLI(LightningCLI):
    def before_instantiate_classes(self):

        torch.set_float32_matmul_precision("high")
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        # Merge config-level model block. Expect either:
        #  - model.class_path + model.init_args (preferred)
        cfg = getattr(self, "config", {}) or {}
        model_cfg = cfg.get("model") if isinstance(cfg, dict) else None

        decoder_instance = None
        if isinstance(model_cfg, dict) and model_cfg.get("class_path"):
            class_path = model_cfg["class_path"]
            init_args = dict(model_cfg.get("init_args") or {})

            # try to fill vocab info if provided
            vocab_path = model_cfg.get("vocab_path") or cfg.get("data", {}).get("vocab_path")
            if vocab_path:
                try:
                    vocab = LatexVocab.load(vocab_path)
                    init_args.setdefault("vocab_size", len(vocab))
                    init_args.setdefault("pad_idx", vocab.PAD)
                except Exception:
                    pass

            module_name, class_name = class_path.rsplit(".", 1)
            mod = import_module(module_name)
            ModelCls = getattr(mod, class_name)
            decoder_instance = ModelCls(**init_args)

        # attach built decoder into instantiate kwargs so LightningCLI will
        # pass it to HMELightningModule constructor as `model=`.
        if decoder_instance is not None:
            self.instantiate_kwargs = getattr(self, "instantiate_kwargs", {})
            self.instantiate_kwargs["model"] = decoder_instance

    def before_fit(self):
        # compile the inner PyTorch model for faster training/inference
        try:
            self.model.model = torch.compile(self.model.model)  # type: ignore
        except Exception:
            # compilation is optional — don't fail if unsupported
            pass


def main():
    HandToTexCLI(
        model_class=HMELightningModule,
        datamodule_class=HMELightningDataModule,
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":
    main()
