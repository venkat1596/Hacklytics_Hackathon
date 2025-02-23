import os
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import functional as F
import itertools
import pytorch_lightning as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import TED, PatchDiscriminator


class CycleMRIGAN(pl.LightningModule):
    def __init__(self, generator_config):
        super().__init__()
        self.save_hyperparameters()
        self.generator_config = generator_config

        # Generators (using TED architecture)
        self.generator_1_5_to_3 = TED(in_channels=1)
        self.generator_3_to_1_5 = TED(in_channels=1)

        # Discriminators
        self.discriminator_1_5_to_3 = PatchDiscriminator(in_channels=1)
        self.discriminator_3_to_1_5 = PatchDiscriminator(in_channels=1)

        # Loss weights
        self.lambda_cycle = 10.0
        self.lambda_identity = 5.0
        self.automatic_optimization = False

    def adversarial_loss(self, y_hat, y):
        return torch.mean((y_hat - y) ** 2)

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

        schG = torch.optim.lr_scheduler.CosineAnnealingLR(optG,
                                                        T_max=self.generator_config["max_epochs"],
                                                        eta_min=self.generator_config["min_lr"])
        schD = torch.optim.lr_scheduler.CosineAnnealingLR(optD,
                                                        T_max=self.generator_config["max_epochs"],
                                                        eta_min=self.generator_config["min_lr"])
        return [optG, optD], [schG, schD]

    def generator_step(self, mri_img_1_5, mri_img_3):
        # Identity loss
        idt_3 = self.generator_1_5_to_3(mri_img_3)
        idt_1_5 = self.generator_3_to_1_5(mri_img_1_5)
        identity_loss = (
                                F.l1_loss(idt_1_5, mri_img_1_5) +
                                F.l1_loss(idt_3, mri_img_3)
                        ) * self.lambda_identity

        # GAN forward
        fake_3 = self.generator_1_5_to_3(mri_img_1_5)
        fake_1_5 = self.generator_3_to_1_5(mri_img_3)

        with torch.no_grad():
            disc_fake_3 = self.discriminator_1_5_to_3(fake_3)
            disc_fake_1_5 = self.discriminator_3_to_1_5(fake_1_5)

        # Cycle consistency
        cycle_1_5 = self.generator_3_to_1_5(fake_3)
        cycle_3 = self.generator_1_5_to_3(fake_1_5)

        cycle_loss = (
                             F.l1_loss(cycle_1_5, mri_img_1_5) +
                             F.l1_loss(cycle_3, mri_img_3)
                     ) * self.lambda_cycle

        # Generator adversarial loss
        g_loss = (
                self.adversarial_loss(disc_fake_3, torch.ones_like(disc_fake_3) * 0.9) +
                self.adversarial_loss(disc_fake_1_5, torch.ones_like(disc_fake_1_5) * 0.9)
        )

        total_g_loss = g_loss + cycle_loss + identity_loss

        return {
            "loss": total_g_loss,
            "g_loss": g_loss,
            "cycle_loss": cycle_loss,
            "identity_loss": identity_loss
        }

    def noisy_labels(self, size, target_value):
        # Add noise to the target labels and clamp to valid range
        noisy = torch.ones_like(size) * target_value + torch.randn_like(size) * 0.05
        return torch.clamp(noisy, 0.0, 1.0)

    def discriminator_step(self, mri_img_1_5, mri_img_3):
        # Generate fakes
        with torch.no_grad():
            fake_3 = self.generator_1_5_to_3(mri_img_1_5)
            fake_1_5 = self.generator_3_to_1_5(mri_img_3)

        # Real loss
        disc_real_3 = self.discriminator_1_5_to_3(mri_img_3)
        disc_real_1_5 = self.discriminator_3_to_1_5(mri_img_1_5)
        real_loss = (
                self.adversarial_loss(disc_real_3, self.noisy_labels(disc_real_3, 0.9)) +
                self.adversarial_loss(disc_real_1_5, self.noisy_labels(disc_real_1_5, 0.9))
        )

        # Fake loss
        disc_fake_3 = self.discriminator_1_5_to_3(fake_3.detach())
        disc_fake_1_5 = self.discriminator_3_to_1_5(fake_1_5.detach())
        fake_loss = (
                self.adversarial_loss(disc_fake_3, self.noisy_labels(disc_fake_3, 0.1)) +
                self.adversarial_loss(disc_fake_1_5, self.noisy_labels(disc_fake_1_5, 0.1))
        )

        return {"loss": (real_loss + fake_loss) * 0.5}

    def training_step(self, batch, batch_idx):
        opt_g, opt_d = self.optimizers()
        mri_img_1_5, mri_img_3 = batch["source"], batch["target"]

        # Generator training phase
        self.toggle_optimizer(opt_g)
        g_loss_dict = self.generator_step(mri_img_1_5, mri_img_3)
        self.manual_backward(g_loss_dict["loss"])
        opt_g.step()
        opt_g.zero_grad()
        self.untoggle_optimizer(opt_g)

        # Log generator metrics
        self.log("generator_total_loss", g_loss_dict["loss"], prog_bar=True, on_epoch=True)
        self.log("generator_loss_G", g_loss_dict["g_loss"], prog_bar=True, on_epoch=True)
        self.log("generator_cycle_loss", g_loss_dict["cycle_loss"], prog_bar=True, on_epoch=True)
        self.log("generator_identity_loss", g_loss_dict["identity_loss"], prog_bar=True, on_epoch=True)

        # Discriminator training phase
        self.toggle_optimizer(opt_d)
        d_loss_dict = self.discriminator_step(mri_img_1_5, mri_img_3)
        self.manual_backward(d_loss_dict["loss"])
        opt_d.step()
        opt_d.zero_grad()
        self.untoggle_optimizer(opt_d)

        self.log("discriminator_total_loss", d_loss_dict["loss"], prog_bar=True, on_epoch=True)

    def on_train_epoch_end(self):
        # Log current learning rates
        opt_g, opt_d = self.optimizers()
        sch_g, sch_d = self.lr_schedulers()
        sch_g.step()
        sch_d.step()

        # Log LR values
        self.log("lr_g", opt_g.param_groups[0]['lr'])
        self.log("lr_d", opt_d.param_groups[0]['lr'])

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
        os.makedirs('./CycleGAN/visualizations', exist_ok=True)
        fig.write_html(f'./CycleGAN/visualizations/comp_s{str(step).zfill(3)}_b{str(batch_idx).zfill(3)}.html')
        return fig

    def validation_step(self, batch, batch_idx):
        mri_img_1_5, mri_img_3 = batch["source"], batch["target"]
        loss_dict = self.generator_step(mri_img_1_5, mri_img_3)

        # Log validation metrics
        self.log("val_generator_total_loss", loss_dict["loss"], prog_bar=True, on_epoch=True)
        self.log("val_generator_loss_G", loss_dict["g_loss"], prog_bar=True, on_epoch=True)
        self.log("val_generator_cycle_loss", loss_dict["cycle_loss"], prog_bar=True, on_epoch=True)
        self.log("val_generator_identity_loss", loss_dict["identity_loss"], prog_bar=True, on_epoch=True)

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

        return loss_dict["loss"]