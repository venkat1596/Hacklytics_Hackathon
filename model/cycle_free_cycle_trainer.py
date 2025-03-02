from pathlib import Path
from typing import Any

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import OptimizerLRScheduler, STEP_OUTPUT

from . import MultiScaleFusionGenerator, PatchSampleF, NLayerDiscriminator
from .cyclefreeGAN_generator import InvertibleGenerator
from .cyclefreeGAN_discriminator import Discriminator_patch as Discriminator


class PatchNCELoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool

    def forward(self, feat_q, feat_k):
        num_patches = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_k = feat_k.detach()

        # pos logit
        l_pos = torch.bmm(
            feat_q.view(num_patches, 1, -1), feat_k.view(num_patches, -1, 1))
        l_pos = l_pos.view(num_patches, 1)

        # neg logit

        # Should the negatives from the other samples of a minibatch be utilized?
        # In CUT and FastCUT, we found that it's best to only include negatives
        # from the same image. Therefore, we set
        # --nce_includes_all_negatives_from_minibatch as False
        # However, for single-image translation, the minibatch consists of
        # crops from the "same" high-resolution image.
        # Therefore, we will include the negatives from the entire minibatch.
        include_negative = self.opt.get('nce_includes_all_negatives_from_minibatch', False)

        if include_negative:
            # reshape features as if they are all negatives of minibatch of size 1.
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = self.opt['batch_size']

        # reshape features to batch size
        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        # diagonal entries are similarity between same features, and hence meaningless.
        # just fill the diagonal with very small number, which is exp(-10) and almost zero
        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        out = torch.cat((l_pos, l_neg), dim=1) / self.opt['nce_T']

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))

        return loss


class CycleFreeContrastiveTraining(pl.LightningModule):
    def __init__(self, generator_config, discriminator_config):
        super().__init__()
        self.save_hyperparameters()
        self.generator_config = generator_config
        self.discriminator_config = discriminator_config

        if self.generator_config["model"] == "multi_scale_fusion":
            self.generator = MultiScaleFusionGenerator(input_nc=generator_config["in_channels"],
                                                       output_nc=generator_config["out_channels"],
                                                       ngf=generator_config["features"], n_blocks=generator_config["n_blocks"])
        elif self.generator_config["model"] == "invertible":
            self.generator = InvertibleGenerator(in_channel=generator_config["in_channel"], 
                                                 n_block=generator_config["n_block"],
                                                 squeeze_num=generator_config["squeeze_num"], 
                                                 conv_lu=generator_config["conv_lu"], block_type=generator_config["block_type"])
        else:
            raise ValueError("Generator model not supported")

        if generator_config["f_model"] == "PatchSampleF":
            self.f_model = PatchSampleF(use_mlp=generator_config["use_mlp"], init_type=generator_config["init_type"],
                                        init_gain=generator_config["init_gain"], nc=generator_config["nc"],
                                        gpu_ids=generator_config["gpu_ids"], num_layers=len(generator_config["layers"]))
        else:
            raise ValueError("F model not supported")


        if self.discriminator_config["model"] == "CUT":
            self.discriminator = NLayerDiscriminator(input_nc=discriminator_config["in_channels"],
                                                     ndf=discriminator_config["features"],
                                                     n_layers=discriminator_config["n_layers"])
        elif self.discriminator_config["model"] == "cyclefreeGAN":
            self.discriminator = Discriminator()
        else:
            raise ValueError("Discriminator model not supported")

        # loss function
        self.l1_loss = nn.L1Loss()
        self.criterionNCE = []
        for n_layer in range(generator_config["n_layers"]):
            setattr(self, f"patchnce_{n_layer}", PatchNCELoss(self.generator_config))
            self.criterionNCE.append(getattr(self, f"patchnce_{n_layer}"))

        self.automatic_optimization = False
        self.identity_loss = nn.MSELoss()


    def configure_optimizers(self):
        optG = torch.optim.Adam(self.generator.parameters(), lr=self.generator_config["lr"], betas=(0.5, 0.999))
        optF = torch.optim.Adam(self.f_model.parameters(), lr=self.generator_config["f_lr"], betas=(0.5, 0.999))
        optD = torch.optim.Adam(self.discriminator.parameters(), lr=self.discriminator_config["lr"], betas=(0.5, 0.999))

        schG = torch.optim.lr_scheduler.CosineAnnealingLR(optG, T_max=self.generator_config["max_epochs"], eta_min=self.generator_config["min_lr"])
        schF = torch.optim.lr_scheduler.CosineAnnealingLR(optF, T_max=self.generator_config["max_epochs"], eta_min=self.generator_config["min_lr"])
        schD = torch.optim.lr_scheduler.CosineAnnealingLR(optD, T_max=self.discriminator_config["max_epochs"], eta_min=self.discriminator_config["min_lr"])

        return [optG, optF, optD], [schG, schF, schD]

    def forward(self, x):
        return self.generator(x, encode_only=False)

    def set_require_grad(self, train_mode):
        if train_mode == "gen":
            for param in self.generator.parameters():
                param.requires_grad = True
            for param in self.discriminator.parameters():
                param.requires_grad = False

        else:
            for param in self.generator.parameters():
                param.requires_grad = False
            for param in self.discriminator.parameters():
                param.requires_grad = True

    def generator_step(self, single_image, image_size, train_mode="train"):
        self.set_require_grad("gen")

        # Split input for clarity
        real_target = single_image[image_size:]

        # Forward pass (generate both target and identity in one step)
        fake_image = self.generator(single_image, encode_only=False)
        fake_image_target = fake_image[:image_size]
        fake_image_identity = fake_image[image_size:]

        # Encode fake images and split features in one pass
        fake_encoding = self.generator(fake_image, layers=self.generator_config['layers'], encode_only=True)
        fake_target_encoded = [feat[:image_size] for feat in fake_encoding]
        fake_identity_encoded = [feat[image_size:] for feat in fake_encoding]

        # Get real features
        real_features_encoded = self.generator(real_target, layers=self.generator_config['layers'], encode_only=True)

        # Apply F-net to extract patches
        real_features_patched, patch_ids = self.f_model(real_features_encoded,
                                                        num_patches=self.generator_config["num_patches"])
        fake_identity_patched, _ = self.f_model(fake_identity_encoded, num_patches=self.generator_config["num_patches"],
                                                patch_ids=patch_ids)
        fake_target_patched, _ = self.f_model(fake_target_encoded, num_patches=self.generator_config["num_patches"],
                                              patch_ids=patch_ids)

        # Calculate NCE losses in a single loop
        n_layers = len(self.generator_config['layers'])
        total_nce_loss_identity = 0.0
        total_nce_loss_encoded = 0.0

        for real_feat, fake_id_feat, fake_tgt_feat, crit, nce_layer in zip(
                real_features_patched, fake_identity_patched, fake_target_patched,
                self.criterionNCE, self.generator_config['layers']):
            # Identity NCE loss
            identity_loss = crit(real_feat, fake_id_feat) * self.generator_config['nce_weight_identity']
            total_nce_loss_identity += identity_loss.mean()

            # Target NCE loss
            target_loss = crit(real_feat, fake_tgt_feat) * self.generator_config['nce_weight']
            total_nce_loss_encoded += target_loss.mean()

        total_nce_loss_identity /= n_layers
        total_nce_loss_encoded /= n_layers

        # Calculate generator and identity losses
        pred_fake_target = self.discriminator(fake_image_target)
        loss_G_fake = self.l1_loss(pred_fake_target, torch.ones_like(pred_fake_target) * 0.9)
        loss_G_identity = self.identity_loss(fake_image_identity, real_target)

        # Combine all losses
        total_loss = loss_G_fake + loss_G_identity * 5.0 + total_nce_loss_identity + total_nce_loss_encoded

        # Logging
        self.log(f"{train_mode}_total_loss", total_loss, prog_bar=True, on_epoch=True)
        self.log(f"{train_mode}_g_loss", loss_G_fake, prog_bar=True, on_epoch=True)
        self.log(f"{train_mode}_identity_loss", loss_G_identity, prog_bar=True, on_epoch=True)
        self.log(f"{train_mode}_nce_loss_identity", total_nce_loss_identity, prog_bar=True, on_epoch=True)
        self.log(f"{train_mode}_nce_loss_encoded", total_nce_loss_encoded, prog_bar=True, on_epoch=True)

        return total_loss

    def discriminator_step(self, single_image, image_size, train_mode="train"):
        self.set_require_grad("disc")
        # Forward pass
        fake_image = self.generator(single_image, encode_only=False)
        fake_image_target = fake_image[:image_size]

        pred_fake_target = self.discriminator(fake_image_target)
        loss_D_fake = self.l1_loss(pred_fake_target, torch.ones_like(pred_fake_target) * 0.1)

        pred_real = self.discriminator(single_image[image_size:])
        loss_D_real = self.l1_loss(pred_real, torch.ones_like(pred_real) * 0.9)

        total_loss = loss_D_fake + loss_D_real
        self.log(f"{train_mode}_d_loss", total_loss, prog_bar=True, on_epoch=True)
        return total_loss

    def training_step(self, batch, batch_idx):
        source, target = batch["source"], batch["target"]
        image_size = source.shape[0]
        single_image = torch.cat((source, target), dim=0)
        opt_g, opt_f, opt_d = self.optimizers()

        # Get accumulation and clipping values from config
        accumulate_steps = self.generator_config['accumulate_grad_batches']
        clip_val = self.generator_config['gradient_clip_val']

        # === GENERATOR PHASE ===
        self.toggle_optimizer(opt_g)
        self.toggle_optimizer(opt_f)

        # Scale loss by 1/accumulate_steps for proper gradient accumulation
        g_loss = self.generator_step(single_image, image_size, train_mode="train") / accumulate_steps
        self.manual_backward(g_loss)

        # Only update weights after accumulating gradients for specified steps
        if (batch_idx + 1) % accumulate_steps == 0:
            # Apply gradient clipping before optimizer step
            # self.clip_gradients(opt_g, gradient_clip_val=clip_val, gradient_clip_algorithm="norm")
            # self.clip_gradients(opt_f, gradient_clip_val=clip_val, gradient_clip_algorithm="norm")

            # Perform optimizer steps
            opt_g.step()
            opt_g.zero_grad()
            opt_f.step()
            opt_f.zero_grad()

        self.untoggle_optimizer(opt_g)
        self.untoggle_optimizer(opt_f)

        # === DISCRIMINATOR PHASE ===
        self.toggle_optimizer(opt_d)

        # Scale loss for gradient accumulation
        d_loss = self.discriminator_step(single_image, image_size, train_mode="train") / accumulate_steps
        self.manual_backward(d_loss)

        # Only update weights after accumulating for specified steps
        if (batch_idx + 1) % accumulate_steps == 0:
            # Apply gradient clipping
            self.clip_gradients(opt_d, gradient_clip_val=clip_val, gradient_clip_algorithm="norm")

            # Perform optimizer step
            opt_d.step()
            opt_d.zero_grad()

        self.untoggle_optimizer(opt_d)

        # Return original losses (unscaled) for logging
        return {"g_loss": g_loss * accumulate_steps, "d_loss": d_loss * accumulate_steps}

    def on_train_epoch_end(self):
        # Log current learning rates
        opt_g, opt_f, opt_d = self.optimizers()
        sch_g, sch_f, sch_d = self.lr_schedulers()
        sch_g.step()
        sch_f.step()
        sch_d.step()

        # Log LR values
        self.log("lr_g", opt_g.param_groups[0]['lr'])
        self.log("lr_f", opt_f.param_groups[0]['lr'])
        self.log("lr_d", opt_d.param_groups[0]['lr'])

    def denormalization(self, x):
        x = (x * 0.5) + 0.5
        return x

    def validation_step(self, batch, batch_idx):
        source, target = batch["source"], batch["target"]
        image_size = source.shape[0]
        single_image = torch.cat((source, target), dim=0)

        # Generator training phase
        g_loss = self.generator_step(single_image, image_size, train_mode="val")

        # Discriminator training phase
        d_loss = self.discriminator_step(single_image, image_size, train_mode="val")

        if self.global_step % 5 == 0:
            # Get generated image from the forward pass
            with torch.no_grad():
                generated_image = self.generator(source, encode_only=False)
                generated_image = self.denormalization(generated_image)
                source = self.denormalization(source)
                target = self.denormalization(target)

                image_size = source.shape[0]
                for image_index in range(image_size):
                    generated_image_tmp = generated_image[image_index]
                    source_tmp = source[image_index]
                    target_tmp = target[image_index]

                    generated_image_tmp = generated_image_tmp.permute(1, 2, 0).cpu().numpy()
                    source_tmp = source_tmp.permute(1, 2, 0).cpu().numpy()
                    target_tmp = target_tmp.permute(1, 2, 0).cpu().numpy()
                    image_name = batch["image_name"][image_index]
                    self.visualize_and_save_comparison(
                        source=source_tmp,
                        target=target_tmp,
                        generated_image=generated_image_tmp,
                        step=self.global_step,
                        image_name=image_name,
                    )
        return g_loss

    def visualize_and_save_comparison(self, source, target, generated_image, image_name, step):
        """Create and save visualization comparing input, generated, and target images."""
        save_dir = Path(self.generator_config["save_dir"]) / image_name
        save_dir.mkdir(exist_ok=True, parents=True)

        source_path = save_dir / f"{step}_source.jpg"
        target_path = save_dir / f"{step}_target.jpg"
        generated_path = save_dir / f"{step}_generated.jpg"

        source = (source * 255).astype('uint8')
        target = (target * 255).astype('uint8')
        generated_image = (generated_image * 255).astype('uint8')

        cv2.imwrite(str(source_path), source)
        cv2.imwrite(str(target_path), target)
        cv2.imwrite(str(generated_path), generated_image)









