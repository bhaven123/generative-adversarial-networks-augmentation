# Implementation adapted from
# https://github.com/PyTorchLightning/lightning-bolts/blob/0.3.0/pl_bolts/models/gans/dcgan/dcgan_module.py#L21-L173
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pytorch_lightning import LightningModule
from torch.nn import BCELoss
from torchvision.io import write_video

from discriminator import Discriminator2d, Discriminator3d
from generator import Generator2d, Generator3d

ROOT = Path(__file__).resolve().parent
TEST_DIR = ROOT / "results"
if not TEST_DIR.exists():
    os.makedirs(TEST_DIR, exist_ok=True)


class DCGAN(LightningModule):
    def __init__(
        self,
        beta1: float = 0.5,
        feature_maps_gen: int = 64,
        feature_maps_disc: int = 64,
        image_channels: int = 1,
        latent_dim: int = 100,
        learning_rate: float = 0.0002,
        dimensionality: int = 2,
        label_noise: float = 0,
        label_smoothing: float = 0,
        **kwargs: Any,
    ) -> None:
        """
        Parameters
        ----------
        beta1: float
            Beta1 value for Adam optimizer

        feature_maps_gen: int
            Number of feature maps to use for the generator

        feature_maps_disc: int
            Number of feature maps to use for the discriminator

        image_channels: int
            Number of channels of the images from the dataset

        latent_dim: float
            Dimension of the latent space (size of input noise vector)

        learning_rate: float
            Learning rate

        dimensionality: 4
            To use actual video

        label_noise: float = 0
            Probality of random label being swapped.

        label_smoothing: float = 1
            Maximum size of noise to add to labels. Must be in (0, 1], and adds
            uniform noise of factor 1 / label_smoothing. E.g. if label_smoothing=5,
            then we add torch.rand([1]) / 5 to each label.
        """
        super().__init__(**kwargs)
        self.save_hyperparameters()

        gen_args = dict(in_channels=latent_dim, n_feature_maps=feature_maps_gen, out_channels=image_channels)
        disc_args = dict(in_channels=image_channels, n_feature_maps=feature_maps_disc)
        generator = Generator2d if dimensionality == 2 else Generator3d
        discriminator = Discriminator2d if dimensionality == 2 else Discriminator3d
        self.generator = generator(**gen_args)
        self.generator.apply(self._weights_init)
        self.discriminator = discriminator(**disc_args)
        self.discriminator.apply(self._weights_init)

        self.criterion = BCELoss()

    def configure_optimizers(self):
        lr = self.hparams.learning_rate
        betas = (self.hparams.beta1, 0.999)
        opt_disc = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=betas)
        opt_gen = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=betas)
        return [opt_disc, opt_gen], []

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        """Generates an image given input noise

        Example::
            noise = torch.rand(batch_size, latent_dim)
            gan = GAN.load_from_checkpoint(PATH)
            img = gan(noise)
        """
        if self.hparams.dimensionality == 2:
            noise = noise.view(*noise.shape, 1, 1)
        else:
            noise = noise.view(*noise.shape, 1, 1, 1)
        return self.generator(noise)

    def training_step(self, batch, batch_idx, optimizer_idx):
        real, _ = batch

        # Train discriminator
        result = None
        if optimizer_idx == 0:
            result = self._disc_step(real, batch_idx)

        # Train generator
        if optimizer_idx == 1:
            result = self._gen_step(real)

        return result

    def _disc_step(self, real: torch.Tensor, batch_idx: int) -> torch.Tensor:
        disc_loss = self._get_disc_loss(real)
        self.log("loss/disc", disc_loss, on_epoch=True)
        self.log("disc_loss", disc_loss, prog_bar=True)
        # if disc_loss < 1e-10 and batch_idx > 1000:
        #     raise RuntimeError("Discriminator loss vanished.")
        return disc_loss

    def _gen_step(self, real: torch.Tensor) -> torch.Tensor:
        gen_loss = self._get_gen_loss(real)
        self.log("loss/gen", gen_loss, on_epoch=True)
        self.log("gen_loss", gen_loss, prog_bar=True)
        return gen_loss

    def _get_disc_loss(self, real: torch.Tensor) -> torch.Tensor:
        # Train with real
        real_pred = self.discriminator(real)
        s = self.hparams.label_smoothing
        if s > 0:
            real_gt = torch.ones_like(real_pred) - (torch.rand_like(real_pred) / s)
        else:
            real_gt = torch.ones_like(real_pred)

        real_loss = self.criterion(real_pred, real_gt)

        # Train with fake
        fake_pred = self._get_fake_pred(real)
        if s > 0:
            fake_gt = torch.zeros_like(fake_pred) + (torch.rand_like(fake_pred) / s)
        else:
            fake_gt = torch.zeros_like(fake_pred)
        fake_loss = self.criterion(fake_pred, fake_gt)

        disc_loss = real_loss + fake_loss

        return disc_loss

    def _get_gen_loss(self, real: torch.Tensor) -> torch.Tensor:
        # Train with fake
        fake_pred = self._get_fake_pred(real)
        fake_gt = torch.ones_like(fake_pred)
        gen_loss = self.criterion(fake_pred, fake_gt)

        return gen_loss

    def _get_fake_pred(self, real: torch.Tensor) -> torch.Tensor:
        batch_size = len(real)
        noise = self._get_noise(batch_size, self.hparams.latent_dim)
        fake = self(noise)
        fake_pred = self.discriminator(fake)

        return fake_pred

    def _get_noise(self, n_samples: int, latent_dim: int, device=None) -> torch.Tensor:
        return torch.randn(n_samples, latent_dim, device=self.device if device is None else device)

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--beta1", default=0.5, type=float)
        parser.add_argument("--feature_maps_gen", default=64, type=int)
        parser.add_argument("--feature_maps_disc", default=64, type=int)
        parser.add_argument("--latent_dim", default=100, type=int)
        parser.add_argument("--learning_rate", default=0.0002, type=float)
        return parser

    @staticmethod
    def _weights_init(m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            torch.nn.init.normal_(m.weight, 1.0, 0.02)
            torch.nn.init.zeros_(m.bias)

    def manual_test(self, num_samples: int = 20) -> None:
        dim = (num_samples, self.hparams.latent_dim)
        # z = torch.normal(mean=0.0, std=0.1, size=dim, device="cuda")
        if self.hparams.latent_dim == 1:
            z = torch.linspace(-1, 1, steps=num_samples, device="cuda").reshape(dim)
        else:
            # just generate random points
            z = self._get_noise(num_samples, self.hparams.latent_dim, device="cuda").reshape(dim)
        with torch.no_grad():
            self.cuda()
            self.eval()
            videos = self(z).cpu().numpy()  # (n_samples, 3, 64, 64, 64) == (N, C, T, H, W)
            self.train()
        videos = videos.transpose([0, 2, 3, 4, 1])  # move to channels_last
        vmax, vmin = videos.max(), videos.min()
        videos -= vmin
        videos /= vmax - vmin
        videos *= 255
        videos = np.floor(videos)
        videos = videos.astype(np.uint8)

        for i in range(videos.shape[0]):
            write_video(str(TEST_DIR / f"vid{i}.mkv"), videos[i], fps=30, video_codec="h264")
        print(f"Wrote sample GAN videos to {TEST_DIR}")
