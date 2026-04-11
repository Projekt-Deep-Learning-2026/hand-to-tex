import argparse
import tkinter as tk
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from hand_to_tex.datasets.dataset import _HMEDatasetBase
from hand_to_tex.datasets.ink_data import InkData
from hand_to_tex.models.lit_module import HMELightningModule
from hand_to_tex.utils import logger
from hand_to_tex.utils.interactive import HMEDrawingApp


def run_batch_inference(
    input_path_str: str,
    model: HMELightningModule,
    device: torch.device,
    save_img: bool,
):
    input_path = Path(input_path_str)

    if input_path.is_file():
        inkml_files = [input_path]
    elif input_path.is_dir():
        inkml_files = list(input_path.rglob("*.inkml"))
    else:
        raise FileNotFoundError(f"Invalid input path: {input_path}")

    if not inkml_files:
        logger.warning(f"No .inkml files found in {input_path}")
        return

    for inkml_path in inkml_files:
        logger.info(f"\nProcessing: {inkml_path.name}")
        ink = InkData.load(inkml_path)

        features = _HMEDatasetBase.extract_features(ink)

        if features.size(0) == 0:
            logger.warning("This .inkml file doesn't contain any traces")
            continue

        features_batched = features.unsqueeze(0).to(device)
        lengths = torch.tensor([features.size(0)], dtype=torch.long, device=device)

        with torch.inference_mode():
            generated_tokens = model.generate(src=features_batched, src_lengths=lengths)

        predicted_expr = model._to_expr(generated_tokens[0]).replace(" ", "")

        expected_expr = ink.tex_norm

        logger.info("=" * 50)
        logger.info(f"Sample ID: {ink.sample_id}")
        logger.info(f"Real TeX:  {expected_expr}")
        logger.info(f"Predicted: {predicted_expr}")
        logger.info("=" * 50)

        fig, ax = ink.to_fig()

        new_title = (
            f"Sample: {ink.sample_id} ({ink.tag})\n"
            f"Expected: ${expected_expr}$\n"
            f"Predicted: ${predicted_expr}$"
        )
        title_color = "darkgreen" if expected_expr == predicted_expr else "darkred"

        ax.set_title(new_title, color=title_color, pad=15)

        if save_img:
            out_filename = f"pred_{ink.sample_id}.png"
            fig.savefig(out_filename, bbox_inches="tight", dpi=150)
            logger.info(f"Visualization saved to {out_filename}")
            plt.close(fig)
        else:
            plt.show()


def main():
    parser = argparse.ArgumentParser(description="HME Demo: Batch Inference or Interactive App")
    parser.add_argument("--ckpt", type=str, required=True, help="checkpoint .ckpt file path")
    parser.add_argument("--input", type=str, help="Path to .inkml file or directory")
    parser.add_argument(
        "--interactive", action="store_true", help="Launch interactive drawing canvas"
    )
    parser.add_argument(
        "--vocab", type=str, default="data/assets/vocab.json", help="Vocabulary .json file path"
    )
    parser.add_argument("--save-img", action="store_true", help="Save the visualization to a .png")
    args = parser.parse_args()

    if not args.interactive and not args.input:
        parser.error("--input is required unless --interactive is used.")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    logger.info(f"Initialising inference on: {device}")
    logger.info(f"Loading model weights from: {args.ckpt}")

    model = HMELightningModule.load_from_checkpoint(
        checkpoint_path=args.ckpt,
        vocab_path=args.vocab,
        map_location=device,
        pretrained_model_path=None,
        weights_only=True,
    )
    model.eval()
    model.to(device)

    if args.interactive:
        logger.info("Launching interactive mode...")
        root = tk.Tk()
        _app = HMEDrawingApp(root, model, device)
        root.mainloop()
    else:
        run_batch_inference(args.input, model, device, args.save_img)


if __name__ == "__main__":
    main()
