import lightning.pytorch as pl
import matplotlib.pyplot as plt
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from tex_gen.datasets.utils import NUM_FEATURES, draw_features
from tex_gen.types import Features


class DiffusionVisualisationCallback(pl.Callback):
    def __init__(
        self,
        seq_len_range: tuple[int, int],
        every_n_epochs: int,
        image_count: int,
        show_steps: list[int],
    ):
        super().__init__()

        self.seq_range = seq_len_range
        self.every_n_epochs = every_n_epochs
        self.image_count = image_count
        self.show_steps = sorted(show_steps, reverse=True)

    @staticmethod
    def _display_samples(states: dict[int, list[Features]], num_images: int, epoch: int) -> Figure:
        fig, axs = plt.subplots(
            ncols=len(states),
            nrows=num_images,
            figsize=(4 * len(states), 4 * num_images),
            squeeze=False,
        )

        for x_ax, stamp in enumerate(sorted(states.keys(), reverse=True)):
            images = states[stamp]
            for y_ax, img in enumerate(images):
                ax: Axes = axs[y_ax, x_ax]
                draw_features(fts=img, ax=axs[y_ax, x_ax])

                if y_ax == 0:
                    ax.set_title(f"Step $t={stamp}$")

        fig.suptitle(f"Generation | epoch={epoch}", fontsize=16)
        fig.tight_layout()

        return fig

    @torch.inference_mode()
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:

        if trainer.current_epoch % self.every_n_epochs != 0:
            return
        pl_module.eval()

        lengths = torch.randint(
            low=self.seq_range[0],
            high=self.seq_range[1],
            size=(self.image_count,),
            device=pl_module.device,
        )
        L = lengths.max().item()

        x = torch.randn(self.image_count, int(L), NUM_FEATURES, device=pl_module.device)
        positions = (
            torch.arange(L, device=pl_module.device).unsqueeze(0).expand(self.image_count, int(L))
        )
        mask = positions < lengths.unsqueeze(1)

        x = x * mask.unsqueeze(-1).float()

        states: dict[int, list[Features]] = {}
        for t in pl_module.noise_scheduler.timesteps:  # type: ignore
            t_tensor = torch.tensor([t], device=pl_module.device)

            residual = pl_module.model(x, t_tensor, src_key_padding_mask=~mask)  # type: ignore
            x = pl_module.noise_scheduler.step(residual, t, x).prev_sample  # type: ignore

            x = x * mask.unsqueeze(-1).float()

            t_val = t.item()
            if t_val in self.show_steps:
                cpu_x = x.detach().cpu().clone()
                fts = [cpu_x[i, : lengths[i].item()] for i in range(self.image_count)]

                states[t_val] = fts

        _fig = self._display_samples(
            states=states, num_images=self.image_count, epoch=trainer.current_epoch
        )
        plt.show()

        pl_module.train()
