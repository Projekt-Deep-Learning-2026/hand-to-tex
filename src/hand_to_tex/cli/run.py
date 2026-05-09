import os
from functools import lru_cache

import torch
from lightning.pytorch.cli import LightningArgumentParser, LightningCLI

from hand_to_tex.datasets.datamodule import HMELightningDataModule
from hand_to_tex.models.lit_module import HMELightningModule
from hand_to_tex.utils import LatexVocab, logger


class HandToTexCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:

        parser.link_arguments(
            source="model.vocab_path",
            target="model.model.init_args.vocab_size",
            compute_fn=self._vocab_size,
        )
        parser.link_arguments(
            source="model.vocab_path",
            target="model.model.init_args.pad_idx",
            compute_fn=self._pad_idx,
        )

    def before_instantiate_classes(self):
        torch.set_float32_matmul_precision("high")
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    def before_fit(self):
        try:
            logger.info("Starting torch.compile...")
            self.model.model = torch.compile(self.model.model)  # type: ignore
        except Exception as e:
            logger.warning(f"Error during compilation: {e}, continuing on uncompiled model")
            pass

    @staticmethod
    @lru_cache(maxsize=1)
    def _vocab(vocab_path: str) -> LatexVocab:
        return LatexVocab.load(path=vocab_path)

    @classmethod
    def _pad_idx(cls, vocab_path: str) -> int:
        return cls._vocab(vocab_path).PAD

    @classmethod
    def _vocab_size(cls, vocab_path: str) -> int:
        return len(cls._vocab(vocab_path))


def main():
    HandToTexCLI(
        model_class=HMELightningModule,
        datamodule_class=HMELightningDataModule,
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":
    main()
