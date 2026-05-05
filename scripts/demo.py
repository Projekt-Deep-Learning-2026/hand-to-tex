import argparse
import math
import tkinter as tk
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from hand_to_tex.datasets.dataset import _HMEDatasetBase
from hand_to_tex.datasets.ink_data import InkData
from hand_to_tex.models.components import (
    ExperimentalTransformer,
    ExperimentalTransformerKVCache,
)
from hand_to_tex.models.lit_module import HMELightningModule
from hand_to_tex.utils import LatexVocab, logger
from hand_to_tex.utils.interactive import HMEDrawingApp


def _collect_inkml_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return list(input_path.rglob("*.inkml"))
    raise FileNotFoundError(f"Invalid input path: {input_path}")


def _predict_expression(
    ink: InkData,
    model: HMELightningModule,
    device: torch.device,
) -> str:
    features = _HMEDatasetBase.extract_features(ink)

    if features.size(0) == 0:
        raise ValueError("This .inkml file doesn't contain any traces")

    features_batched = features.unsqueeze(0).to(device)
    conv1 = getattr(model.model, "conv1", None)
    in_ch = conv1.weight.shape[1] if conv1 is not None else features_batched.shape[2]
    if features_batched.shape[2] > in_ch:
        features_batched = features_batched[:, :, :in_ch]
    lengths = torch.tensor([features.size(0)], dtype=torch.long, device=device)

    with torch.inference_mode():
        generated_tokens = model.generate(src=features_batched, src_lengths=lengths)

    return model._to_expr(generated_tokens[0]).replace(" ", "")


def _draw_ink(ax: Axes, ink: InkData) -> None:
    for trace in ink.traces:
        if not trace:
            continue
        x_coords = [point[0] for point in trace]
        y_coords = [point[1] for point in trace]
        ax.plot(x_coords, y_coords, color="black", linewidth=2.0)

    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    ax.invert_yaxis()


def _sample_title(ink: InkData, expected_expr: str, predicted_expr: str, use_tex: bool) -> str:
    if use_tex:
        return (
            f"Sample: {ink.sample_id} ({ink.tag})\n"
            f"Expected: ${expected_expr}$\n"
            f"Predicted: ${predicted_expr}$"
        )

    return (
        f"Sample: {ink.sample_id} ({ink.tag})\n"
        f"Expected: {expected_expr}\n"
        f"Predicted: {predicted_expr}"
    )


def _apply_title_with_fallback(
    ax: Axes,
    ink: InkData,
    expected_expr: str,
    predicted_expr: str,
) -> None:
    title_color = "darkgreen" if expected_expr == predicted_expr else "darkred"
    title_tex = _sample_title(ink, expected_expr, predicted_expr, use_tex=True)
    title_plain = _sample_title(ink, expected_expr, predicted_expr, use_tex=False)

    title_obj = ax.set_title(label=title_tex, color=title_color, pad=15)
    try:
        ax.figure.canvas.draw()
    except Exception:
        logger.warning(f"Falling back to plain text title for {ink.sample_id}")
        title_obj = ax.set_title(label=title_plain, color=title_color, pad=15)
        title_obj.set_usetex(False)


def _render_single_sample(
    ink: InkData,
    expected_expr: str,
    predicted_expr: str,
) -> Figure:
    fig, ax = plt.subplots(num=ink.sample_id, figsize=(6, 4))
    _draw_ink(ax, ink)
    _apply_title_with_fallback(ax, ink, expected_expr, predicted_expr)
    fig.tight_layout()
    return fig


def _render_grid(
    samples: list[tuple[InkData, str, str]],
    figure_index: int,
    samples_per_figure: int,
) -> Figure:
    cols = math.ceil(math.sqrt(samples_per_figure))
    rows = math.ceil(samples_per_figure / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))

    axes_flat = [axes] if not hasattr(axes, "flat") else list(axes.flat)

    for axis in axes_flat[len(samples) :]:
        axis.axis("off")

    for axis, (ink, expected_expr, predicted_expr) in zip(axes_flat, samples, strict=False):
        _draw_ink(axis, ink)
        _apply_title_with_fallback(axis, ink, expected_expr, predicted_expr)

    fig.suptitle(f"HME Demo samples {figure_index + 1}", y=0.98)
    fig.tight_layout()
    return fig


def _show_or_save_figure(fig: Figure, save_img: bool, out_filename: str | None) -> None:
    if save_img:
        if out_filename is None:
            raise ValueError("An output file name is required when saving figures")
        fig.savefig(out_filename, bbox_inches="tight", dpi=150)
        logger.info(f"Visualization saved to {out_filename}")
        plt.close(fig)
    else:
        plt.show()


def _load_hparams(ckpt_path: Path) -> dict:
    if not ckpt_path.exists():
        return {}
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hparams = ckpt.get("hyper_parameters", {})
    state_dict = ckpt.get("state_dict", {})
    for key in state_dict.keys():
        if key.endswith("conv1.weight"):
            hparams["in_channels"] = state_dict[key].shape[1]
            break
    return hparams


def _build_model(vocab: LatexVocab, hparams: dict):
    use_kvcache = hparams.get("use_kvcache", False)
    model_cls = ExperimentalTransformerKVCache if use_kvcache else ExperimentalTransformer
    return model_cls(
        in_channels=hparams.get("in_channels", 12),
        vocab_size=len(vocab),
        pad_idx=vocab.PAD,
        d_model=hparams.get("d_model", 256),
        nhead=hparams.get("nhead", 8),
        num_encoder_layers=hparams.get("num_encoder_layers", 4),
        num_decoder_layers=hparams.get("num_decoder_layers", 4),
        dim_feedforward=hparams.get("dim_feedforward", 1024),
        dropout=hparams.get("dropout", 0.1),
    )


def run_batch_inference(
    input_path_str: str,
    model: HMELightningModule,
    device: torch.device,
    save_img: bool,
    samples_per_figure: int,
):
    input_path = Path(input_path_str)

    inkml_files = sorted(_collect_inkml_files(input_path))

    if not inkml_files:
        logger.warning(f"No .inkml files found in {input_path}")
        return
    else:
        logger.info(f"Processing {input_path}, found {len(inkml_files)} .inkml files to process")

    rendered_samples: list[tuple[InkData, str, str]] = []

    for inkml_path in inkml_files:
        logger.info(f"\nProcessing: {inkml_path.name}")
        ink = InkData.load(inkml_path)

        try:
            predicted_expr = _predict_expression(ink, model, device)
        except ValueError as exc:
            logger.warning(str(exc))
            continue

        expected_expr = ink.tex_norm

        logger.info("=" * 50)
        logger.info(f"Sample ID: {ink.sample_id}")
        logger.info(f"Real TeX:  {expected_expr}")
        logger.info(f"Predicted: {predicted_expr}")
        logger.info("=" * 50)

        rendered_samples.append((ink, expected_expr, predicted_expr))

    if not rendered_samples:
        return

    for figure_index, start in enumerate(range(0, len(rendered_samples), samples_per_figure)):
        chunk = rendered_samples[start : start + samples_per_figure]

        if len(chunk) == 1:
            ink, expected_expr, predicted_expr = chunk[0]
            fig = _render_single_sample(ink, expected_expr, predicted_expr)
            out_filename = f"pred_{ink.sample_id}.png" if save_img else None
        else:
            fig = _render_grid(chunk, figure_index, samples_per_figure)
            first_sample_id = chunk[0][0].sample_id
            last_sample_id = chunk[-1][0].sample_id
            out_filename = (
                f"pred_grid_{first_sample_id}_to_{last_sample_id}.png" if save_img else None
            )

        _show_or_save_figure(fig, save_img, out_filename)


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
    parser.add_argument(
        "--samples-per-figure",
        type=int,
        default=1,
        help="Number of samples to render per figure window or saved image",
    )
    parser.add_argument("--save-img", action="store_true", help="Save the visualization to a .png")
    args = parser.parse_args()

    if args.samples_per_figure < 1:
        parser.error("--samples-per-figure must be at least 1.")

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

    vocab = LatexVocab.load(args.vocab)
    hparams = _load_hparams(Path(args.ckpt))
    decoder_model = _build_model(vocab, hparams)

    model = HMELightningModule.load_from_checkpoint(
        checkpoint_path=args.ckpt,
        vocab_path=args.vocab,
        model=decoder_model,
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
        run_batch_inference(args.input, model, device, args.save_img, args.samples_per_figure)


if __name__ == "__main__":
    main()
