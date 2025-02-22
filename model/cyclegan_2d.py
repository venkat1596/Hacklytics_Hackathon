import os
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import functional as F
import itertools
import pytorch_lightning as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .generator_2d import EfficientInvertibleGenerator2D
from .discriminator_2d import ProgressivePatchDiscriminator

class CycleGan2D(pl.LightningModule):
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
            self.generator_3_to_1_5 = EfficientInvertibleGenerator2D(
                dim=generator_config["features"],
                n_blocks=generator_config["n_blocks"],
                ffn_scale=generator_config["ffn_scale"]
            )
        else:
            raise NotImplementedError(f"{generator_config['model']} not implemented")

        if discriminator_config["model"] == "patch_gan":
            self.discriminator_1_5_to_3 = ProgressivePatchDiscriminator(
                in_channels=discriminator_config["in_channels"],
                features=discriminator_config["features"],
            )
            self.discriminator_3_to_1_5 = ProgressivePatchDiscriminator(
                in_channels=discriminator_config["in_channels"],
                features=discriminator_config["features"]
            )
        else:
            raise NotImplementedError(f"{discriminator_config['model']} not implemented")

        self.automatic_optimization = False

    def forward(self, mri_img_1_5):
        return self.generator_1_5_to_3(mri_img_1_5)

    def configure_optimizers(self):
        optG = Adam(
            itertools.chain(
                self.generator_1_5_to_3.parameters(),
                self.generator_3_to_1_5.parameters()
            ),
            lr=1e-4, betas=(0.5, 0.999)
        )

        optD = Adam(
            itertools.chain(
                self.discriminator_1_5_to_3.parameters(),
                self.discriminator_3_to_1_5.parameters()
            ),
            lr=1e-4, betas=(0.5, 0.999)
        )

        gamma = lambda epoch: 1 - max(0, epoch + 1 - 100) / 101
        schG = LambdaLR(optG, lr_lambda=gamma)
        schD = LambdaLR(optD, lr_lambda=gamma)
        return [optG, optD], [schG, schD]

    def generator_step(self, mri_img_1_5, mri_img_3):
        # forward pass
        fake_mri_img_3 = self.generator_1_5_to_3(mri_img_1_5)
        fake_mri_img_1_5 = self.generator_3_to_1_5(mri_img_3)

        same_mri_1_5 = self.generator_3_to_1_5(mri_img_1_5)
        same_mri_3 = self.generator_1_5_to_3(mri_img_3)

        cycle_mri_1_5 = self.generator_3_to_1_5(fake_mri_img_3)
        cycle_mri_3 = self.generator_1_5_to_3(fake_mri_img_1_5)

        with torch.no_grad():
            disc_fake_3 = self.discriminator_1_5_to_3(fake_mri_img_3)
            disc_fake_1_5 = self.discriminator_3_to_1_5(fake_mri_img_1_5)

        # Calculate losses
        loss_identity = (
                F.l1_loss(same_mri_1_5, mri_img_1_5) +
                F.l1_loss(same_mri_3, mri_img_3)
        )

        cycle_loss = (
                F.l1_loss(cycle_mri_1_5, mri_img_1_5) +
                F.l1_loss(cycle_mri_3, mri_img_3)
        )

        loss_G = (
                F.mse_loss(disc_fake_3, torch.ones_like(disc_fake_3)) +
                F.mse_loss(disc_fake_1_5, torch.ones_like(disc_fake_1_5))
        )

        loss = (
                loss_G +
                self.generator_config["cycle_weight"] * cycle_loss +
                self.generator_config["identity_weight"] * loss_identity
        )

        return {
            "total_loss": loss,
            "loss_G": loss_G,
            "cycle_loss": cycle_loss,
            "identity_loss": loss_identity
        }

    def discriminator_step(self, mri_img_1_5, mri_img_3):
        with torch.no_grad():
            fake_mri_img_3 = self.generator_1_5_to_3(mri_img_1_5)
            fake_mri_img_1_5 = self.generator_3_to_1_5(mri_img_3)

        disc_real_3 = self.discriminator_1_5_to_3(mri_img_3)
        disc_real_1_5 = self.discriminator_3_to_1_5(mri_img_1_5)

        disc_fake_3 = self.discriminator_1_5_to_3(fake_mri_img_3)
        disc_fake_1_5 = self.discriminator_3_to_1_5(fake_mri_img_1_5)

        loss = (
                F.mse_loss(disc_real_3, torch.ones_like(disc_real_3)) +
                F.mse_loss(disc_real_1_5, torch.ones_like(disc_real_1_5)) +
                F.mse_loss(disc_fake_3, torch.zeros_like(disc_fake_3)) +
                F.mse_loss(disc_fake_1_5, torch.zeros_like(disc_fake_1_5))
        )

        return {"total_loss": loss}

    def training_step(self, batch, batch_idx):
        opt_g, opt_d = self.optimizers()
        mri_img_1_5, mri_img_3 = batch["source"], batch["target"]

        # Generator training phase
        self.toggle_optimizer(opt_g)
        g_loss_dict = self.generator_step(mri_img_1_5, mri_img_3)
        self.manual_backward(g_loss_dict["total_loss"])
        opt_g.step()
        opt_g.zero_grad()
        self.untoggle_optimizer(opt_g)

        # Log generator metrics
        self.log("generator_total_loss", g_loss_dict["total_loss"], prog_bar=True)
        self.log("generator_loss_G", g_loss_dict["loss_G"])
        self.log("generator_cycle_loss", g_loss_dict["cycle_loss"])
        self.log("generator_identity_loss", g_loss_dict["identity_loss"])

        # Discriminator training phase
        self.toggle_optimizer(opt_d)
        d_loss_dict = self.discriminator_step(mri_img_1_5, mri_img_3)
        self.manual_backward(d_loss_dict["total_loss"])
        opt_d.step()
        opt_d.zero_grad()
        self.untoggle_optimizer(opt_d)

        self.log("discriminator_total_loss", d_loss_dict["total_loss"], prog_bar=True)

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
        os.makedirs('visualizations', exist_ok=True)
        fig.write_html(f'visualizations/comp_s{str(step).zfill(3)}_b{str(batch_idx).zfill(3)}.html')
        return fig

    def validation_step(self, batch, batch_idx):
        mri_img_1_5, mri_img_3 = batch["source"], batch["target"]
        loss_dict = self.generator_step(mri_img_1_5, mri_img_3)

        # Log validation metrics
        self.log("val_generator_total_loss", loss_dict["total_loss"])
        self.log("val_generator_loss_G", loss_dict["loss_G"])
        self.log("val_generator_cycle_loss", loss_dict["cycle_loss"])
        self.log("val_generator_identity_loss", loss_dict["identity_loss"])

        # Visualize results periodically
        if self.global_step % 5 == 0:
            with torch.no_grad():
                generated_3t = self.generator_1_5_to_3(mri_img_1_5)
            self.visualize_and_save_comparison(
                batch=batch,
                generated_image=generated_3t,
                batch_idx=batch_idx,
                step=self.global_step
            )

        return loss_dict["total_loss"]