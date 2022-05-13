# Implementation adapted from
# https://github.com/PyTorchLightning/lightning-bolts/blob/8b4d90d7443ac263fd246e2745c267439f4b9274/pl_bolts/callbacks/vision/image_generation.py#L15
import torch
import torchvision
from pytorch_lightning import Callback


class Tensorboard3dGenerativeModelImageSampler(Callback):
    def __init__(self, num_samples: int = 3):
        """
        Generates images and logs to tensorboard.
        Your model must implement the forward function for generation
        Requirements::
            # model must have img_dim arg
            model.img_dim = (1, 28, 28)
            # model forward must work for sampling
            z = torch.rand(batch_size, latent_dim)
            img_samples = your_model(z)
        Example::
            from pl_bolts.callbacks import TensorboardGenerativeModelImageSampler
            trainer = Trainer(callbacks=[TensorboardGenerativeModelImageSampler()])
        """
        super().__init__()
        self.num_samples = num_samples

    def on_epoch_end(self, trainer, pl_module):
        dim = (self.num_samples, pl_module.hparams.latent_dim)
        z = torch.normal(mean=0.0, std=1.0, size=dim, device=pl_module.device)

        # generate images
        with torch.no_grad():
            pl_module.eval()
            images = torch.mean(pl_module(z), dim=-1, keepdim=False)  # this is the only change
            pl_module.train()

        if len(images.size()) == 2:
            img_dim = pl_module.img_dim
            images = images.view(self.num_samples, *img_dim)

        grid = torchvision.utils.make_grid(images)
        str_title = f"{pl_module.__class__.__name__}_images"
        trainer.logger.experiment.add_image(str_title, grid, global_step=trainer.global_step)


# A tiny modification of
# https://github.com/PyTorchLightning/lightning-bolts/blob/8b4d90d7443ac263fd246e2745c267439f4b9274/pl_bolts/callbacks/vision/image_generation.py#L15
# to handle generated video data by just clipping some frames
class TensorboardVideoSampler(Callback):
    def __init__(self, num_samples: int = 3):
        """
        Generates videos and logs some frames (at 0, 10, 30 frames) to tensorboard.
        Your model must implement the forward function for generation
        Requirements::
            # model must have img_dim arg, e.g.
            model.img_dim = (1, 28, 28)
            # model forward must work for sampling
            z = torch.rand(batch_size, latent_dim)
            img_samples = your_model(z)
        """
        super().__init__()
        self.num_samples = num_samples
        self.epoch = 0

    def on_epoch_end(self, trainer, pl_module):
        self.epoch += 1
        if self.epoch % 10 != 0:
            return
        dim = (self.num_samples, pl_module.hparams.latent_dim)
        z = torch.normal(mean=0.0, std=1.0, size=dim, device=pl_module.device)

        # generate images
        with torch.no_grad():
            pl_module.eval()
            images = pl_module(z)  # average along time?
            pl_module.train()

        if len(images.size()) == 2:
            img_dim = pl_module.img_dim
            images = images.view(self.num_samples, *img_dim)

        grid0 = torchvision.utils.make_grid(images[:, :, 0])
        grid10 = torchvision.utils.make_grid(images[:, :, 10])
        grid30 = torchvision.utils.make_grid(images[:, :, 30])
        str_title0 = f"{pl_module.__class__.__name__}_images0"
        str_title1 = f"{pl_module.__class__.__name__}_images1"
        str_title2 = f"{pl_module.__class__.__name__}_images2"
        trainer.logger.experiment.add_image(str_title0, grid0, global_step=trainer.global_step)
        trainer.logger.experiment.add_image(str_title1, grid10, global_step=trainer.global_step)
        trainer.logger.experiment.add_image(str_title2, grid30, global_step=trainer.global_step)
