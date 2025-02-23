from typing import Any
import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .generator_2d import EfficientInvertibleGenerator2D
from .discriminator_2d import SimpleDiscriminator

class CycleFreeCycleGan(pl.LightningModule):
    def __init__(self, generator_config, discriminator_config):
        super().__init__()
        self.save_hyperparameters()
        self.generator_config = generator_config
        self.discriminator_config = discriminator_config

        if generator_config["model"] == "SAFM":
            self.generator_1_5_to_3 = EfficientInvertibleGenerator2D(
                dim=generator_config["features"],
                n_blocks=generator_config["n_blocks"],
                ffn_scale=generator_config["ffn_scale"]
            )

        if discriminator_config["model"] == "patch_gan":
            self.discriminator_1_5_to_3 = SimpleDiscriminator(in_channels=discriminator_config["in_channels"],
                                                              d=discriminator_config["features"])

        # Loss functions with label smoothing
        self.adversarial_loss = nn.MSELoss()
        self.cycle_loss = nn.L1Loss()

        # Gradient clipping
        self.max_grad_norm = 1.0

        self.automatic_optimization = False

    def forward(self, mri_img_1_5):
        return self.generator_1_5_to_3(mri_img_1_5)

    def configure_optimizers(self):
        optG = torch.optim.Adam(
            self.generator_1_5_to_3.parameters(),
            lr=self.generator_config['lr'], betas=(0.5, 0.999)
        )

        optD = torch.optim.Adam(
            self.discriminator_1_5_to_3.parameters(),
            lr=self.discriminator_config['lr'], betas=(0.5, 0.999)
        )

        lr_scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(optG,
                                                                    T_max=self.generator_config["max_epochs"],
                                                                    eta_min=self.generator_config["min_lr"])
        lr_scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(optD,
                                                                    T_max=self.discriminator_config["max_epochs"],
                                                                    eta_min=self.discriminator_config["min_lr"])

        return [optG, optD], [lr_scheduler_G, lr_scheduler_D]

    def _clip_gradients(self):
        """Clip gradients to prevent exploding gradients"""
        torch.nn.utils.clip_grad_norm_(
            self.generator_1_5_to_3.parameters(),
            self.max_grad_norm
        )
        torch.nn.utils.clip_grad_norm_(
            self.discriminator_1_5_to_3.parameters(),
            self.max_grad_norm
        )

    def _prevent_nan(self, tensor, default_value=0.0):
        """Replace NaN values with a default value"""
        return torch.where(
            torch.isnan(tensor),
            torch.full_like(tensor, default_value),
            tensor
        )

    def generator_step(self, mri_img_1_5):
        # Ensure inputs are valid
        mri_img_1_5 = self._prevent_nan(mri_img_1_5)

        # Forward pass
        fake_mri_img_3 = self.generator_1_5_to_3(mri_img_1_5)
        same_mri_1_5 = self.generator_1_5_to_3.inverse(fake_mri_img_3)

        with torch.no_grad():
            disc_fake_3 = self.discriminator_1_5_to_3(fake_mri_img_3)

        g_loss = self.adversarial_loss(disc_fake_3, torch.ones_like(disc_fake_3))
        cycle_loss = self.cycle_loss(same_mri_1_5, mri_img_1_5) * self.generator_config["cycle_weight"]

        total_loss = g_loss + cycle_loss

        return {"total_loss": total_loss, "g_loss": g_loss, "cycle_loss": cycle_loss}

    def discriminator_step(self, mri_img_1_5, mri_img_3):
        # Ensure inputs are valid
        mri_img_1_5 = self._prevent_nan(mri_img_1_5)
        mri_img_3 = self._prevent_nan(mri_img_3)

        # Forward pass
        with torch.no_grad():
            fake_mri_img_3 = self.generator_1_5_to_3(mri_img_1_5)

        disc_real_3 = self.discriminator_1_5_to_3(mri_img_3)
        disc_fake_3 = self.discriminator_1_5_to_3(fake_mri_img_3)

        d_loss = 0.5 * (
            self.adversarial_loss(disc_real_3, torch.ones_like(disc_real_3)) +
            self.adversarial_loss(disc_fake_3, torch.zeros_like(disc_fake_3))
        )

        return {"d_loss": d_loss}

    def visualize_and_save_comparison(self, batch, generated_image, batch_idx, step):
        """Create and save visualization comparing input, generated, and target images."""

        # Get images from batch (shape: B, C, H, W)
        source_image = batch['source'][0, 0].cpu().numpy()
        target_image = batch['target'][0, 0].cpu().numpy()
        generated_image = generated_image[0, 0].detach().cpu().numpy()

        # Denormalize images
        def denormalize(img, global_min, global_max):
            return (img + 1) * (global_max - global_min) / 2 + global_min

        source_image = denormalize(
            source_image,
            batch['source_global_min'].cpu().numpy()[0],
            batch['source_global_max'].cpu().numpy()[0]
        )
        target_image = denormalize(
            target_image,
            batch['target_global_min'].cpu().numpy()[0],
            batch['target_global_max'].cpu().numpy()[0]
        )
        generated_image = denormalize(
            generated_image,
            batch['target_global_min'].cpu().numpy()[0],
            batch['target_global_max'].cpu().numpy()[0]
        )

        # Create figure
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('1.5T Input', 'Generated 3T', 'Ground Truth 3T')
        )

        # Add images
        fig.add_trace(
            go.Heatmap(z=source_image, colorscale='Gray', showscale=False),
            row=1, col=1
        )
        fig.add_trace(
            go.Heatmap(z=generated_image, colorscale='Gray', showscale=False),
            row=1, col=2
        )
        fig.add_trace(
            go.Heatmap(z=target_image, colorscale='Gray', showscale=False),
            row=1, col=3
        )

        # Update layout
        fig.update_layout(
            title=f'MRI Comparison - Step {step}, Batch {batch_idx}',
            height=400,
            width=1200,
            showlegend=False
        )

        # Save visualization
        os.makedirs('two_dim/visualizations', exist_ok=True)
        fig.write_html(f'two_dim/visualizations/comp_s{str(step).zfill(3)}_b{str(batch_idx).zfill(3)}.html')
        return fig

    def training_step(self, batch, batch_idx):
        # Training step
        source, target = batch['source'], batch['target']

        g_opt , d_opt = self.optimizers()

        # Generator training phase
        self.toggle_optimizer(g_opt)
        loss_dict_g = self.generator_step(source)
        self.log("g_loss", loss_dict_g["g_loss"], prog_bar=True, on_epoch=True)
        self.log("cycle_loss", loss_dict_g["cycle_loss"], prog_bar=True, on_epoch=True)
        self.log("total_loss", loss_dict_g["total_loss"], prog_bar=True, on_epoch=True)

        self.manual_backward(loss_dict_g["total_loss"])
        self._clip_gradients()
        g_opt.step()
        g_opt.zero_grad()
        self.untoggle_optimizer(g_opt)

        # Discriminator training phase
        self.toggle_optimizer(d_opt)
        loss_dict_d = self.discriminator_step(source, target)
        self.log("d_loss", loss_dict_d["d_loss"])
        self.manual_backward(loss_dict_d["d_loss"])
        self._clip_gradients()
        d_opt.step()
        d_opt.zero_grad()
        self.untoggle_optimizer(d_opt)

    def on_train_epoch_end(self):
        # Log current learning rates
        opt_g, opt_d = self.optimizers()
        sch_g, sch_d = self.lr_schedulers()
        sch_g.step()
        sch_d.step()

        # Log LR values
        self.log("lr_g", opt_g.param_groups[0]['lr'])
        self.log("lr_d", opt_d.param_groups[0]['lr'])

    def validation_step(self, batch, batch_idx):
        source, target = batch['source'], batch['target']
        loss_dict_g = self.generator_step(source)
        loss_dict_d = self.discriminator_step(source, target)
        self.log("val_g_loss", loss_dict_g["g_loss"], prog_bar=True, on_epoch=True)
        self.log("val_cycle_loss", loss_dict_g["cycle_loss"], prog_bar=True, on_epoch=True)
        self.log("val_total_loss", loss_dict_g["total_loss"], prog_bar=True, on_epoch=True)
        self.log("val_d_loss", loss_dict_d["d_loss"], prog_bar=True, on_epoch=True)

        # Visualize results periodically
        if self.global_step % 5 == 0:
            # Get generated image from the forward pass
            with torch.no_grad():
                generated_3t = self.generator_1_5_to_3(source)
            self.visualize_and_save_comparison(
                batch=batch,
                generated_image=generated_3t,
                batch_idx=batch_idx,
                step=self.global_step
            )
        return loss_dict_g["total_loss"]
