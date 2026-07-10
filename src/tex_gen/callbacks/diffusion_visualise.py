import IPython.display as disp
import lightning.pytorch as pl
import matplotlib.pyplot as plt
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from tex_gen.datasets.utils import NUM_FEATURES, draw_features
from tex_gen.types import Batch, Features


class DiffusionVisualisationCallback(pl.Callback):
    def __init__(
        self,
        seq_len_range: tuple[int, int],
        every_n_epochs: int,
        image_count: int,
        show_steps: list[int],
        show_forward: bool,
        clear_output: bool = True,
    ):
        super().__init__()

        self.seq_range = seq_len_range
        self.every_n_epochs = every_n_epochs
        self.image_count = image_count
        self.show_steps = sorted(show_steps, reverse=True)
        self.show_forward = show_forward
        self.clear_output = clear_output

    @staticmethod
    def _draw_samples(
        states: dict[int, list[Features]], num_images: int, epoch: int, reverse_timesteps: bool
    ) -> Figure:
        fig, axs = plt.subplots(
            ncols=len(states),
            nrows=num_images,
            figsize=(4 * len(states), 4 * num_images),
            squeeze=False,
        )

        for x_ax, stamp in enumerate(sorted(states.keys(), reverse=reverse_timesteps)):
            images = states[stamp]
            for y_ax, img in enumerate(images):
                ax: Axes = axs[y_ax, x_ax]
                draw_features(fts=img, ax=axs[y_ax, x_ax])

                if y_ax == 0:
                    ax.set_title(f"Step $t={stamp}$")

        fig.suptitle(f"Generation | epoch={epoch}", fontsize=16)
        fig.tight_layout()

        return fig

    def display(self, fig: Figure) -> None:
        if self.clear_output:
            disp.clear_output(wait=True)
        plt.show()

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

        fig = self._draw_samples(
            states=states,
            num_images=self.image_count,
            epoch=trainer.current_epoch,
            reverse_timesteps=True,
        )
        self.display(fig=fig)

        pl_module.train()

    @torch.inference_mode()
    def on_train_batch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, batch: Batch, batch_idx: int
    ) -> None:

        if trainer.global_step != 0 or not self.show_forward:
            return

        features, mask = batch

        num_samples = min(self.image_count, features.size(0))
        x0 = features[:num_samples]
        m = mask[:num_samples]

        lengths = m.sum(dim=1).long()
        noise = torch.randn_like(x0) * m.unsqueeze(-1).float()

        states: dict[int, list[Features]] = {}

        for t_val in self.show_steps:
            if t_val == 0:
                noisy_x = x0
            else:
                t_tensor = torch.full(
                    (num_samples,), t_val, device=pl_module.device, dtype=torch.long
                )
                noisy_x = pl_module.noise_scheduler.add_noise(x0, noise, t_tensor)  # type: ignore

            cpu_x = noisy_x.detach().cpu().clone()
            fts = [cpu_x[i, : lengths[i]] for i in range(num_samples)]
            states[t_val] = fts

        fig = self._draw_samples(
            states=states,
            num_images=num_samples,
            epoch=trainer.current_epoch,
            reverse_timesteps=False,
        )
        self.display(fig=fig)
