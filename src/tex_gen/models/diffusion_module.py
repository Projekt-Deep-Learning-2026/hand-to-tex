import lightning.pytorch as pl
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_utils import SchedulerMixin

from tex_gen.models.mse_padding_loss import PaddedMSELoss
from tex_gen.types import Batch


class TexGenLightningModule(pl.LightningModule):
    def __init__(self, backbone: nn.Module, noise_scheduler: SchedulerMixin, lr: float):
        super().__init__()

        self.save_hyperparameters(ignore=["backbone", "noise_scheduler"])

        self.backbone = backbone
        self.noise_scheduler = noise_scheduler
        self.learning_rate = lr

        self.loss_fn = PaddedMSELoss()

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, src_padding_mask: torch.Tensor):

        return self.backbone(x, timesteps, src_padding_mask)

    def _compute_loss(self, batch: Batch) -> torch.Tensor:
        x, mask = batch
        B = x.shape[0]

        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (B,),  # type: ignore
            device=self.device,
        ).long()

        noise = torch.randn_like(x) * mask.unsqueeze(-1).float()

        x_t = self.noise_scheduler.add_noise(x, noise, timesteps)  # type: ignore
        x_t = x_t * mask.unsqueeze(-1).float()

        noise_pred = self.forward(x_t, timesteps, mask)

        match self.noise_scheduler.config.prediction_type:
            case "epsilon":
                target = noise
            case "sample":
                target = x
            case "v prediction":
                target = self.noise_scheduler.get_velocity(x, noise, timesteps)  # type: ignore
            case _:
                raise ValueError(
                    f"Unsupported prediction type: {self.noise_scheduler.config.prediction_type}"
                )

        return self.loss_fn(noise_pred, target, mask)

    def configure_optimizers(self):  # type: ignore
        optimizer = torch.optim.AdamW(self.backbone.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=1000000, eta_min=1e-6
        )
        opt_config = {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
        return opt_config

    @torch.inference_mode()
    def generate(
        self, noise: torch.Tensor, mask: torch.Tensor, show_steps: list[int] | None = None
    ) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:

        self.noise_scheduler.set_timesteps(
            self.noise_scheduler.config.num_train_timesteps, device=self.device
        )  # type: ignore
        x = noise * mask.unsqueeze(-1).float()

        states: dict[int, torch.Tensor] = {}

        for t in self.noise_scheduler.timesteps:
            timesteps = torch.full((x.shape[0],), t, device=self.device, dtype=torch.long)
            noise_pred = self.forward(x, timesteps, mask)
            x = self.noise_scheduler.step(noise_pred, t, x).prev_sample  # type: ignore
            x = x * mask.unsqueeze(-1).float()

            t_val = t.item()
            if show_steps is not None and t_val in show_steps:
                states[t_val] = x.clone()

        return x, states

    def training_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        loss = self._compute_loss(batch=batch)

        self.log("train_loss", loss, prog_bar=True)

        return loss

    @torch.no_grad()
    def validation_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        loss = self._compute_loss(batch=batch)

        self.log("val_loss", loss, prog_bar=True)

        return loss
