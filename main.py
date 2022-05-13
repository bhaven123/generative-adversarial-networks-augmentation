from argparse import ArgumentParser

import numpy as np
from numpy.typing import NDArray
from pl_bolts.callbacks import LatentDimInterpolator, TensorboardGenerativeModelImageSampler
from pytorch_lightning import Trainer
from dcgan import DCGAN
from image_sampler import Tensorboard3dGenerativeModelImageSampler, TensorboardVideoSampler
from loader import get_hmdb_loader

FloatArray = NDArray[np.floating]

DIM_HELP = """
    If --dim=4, then train on the HMDB51 data small training subset.
"""


def test_train() -> None:
    parser = ArgumentParser()
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--dim", default=2, choices=[2, 3, 4], type=int, help=DIM_HELP)
    parser.add_argument(
        "--fmaps", default=64, type=int, help="number of base feature maps in both discriminator and generator"
    )  # feature maps
    parser.add_argument("--label_noise", default=0, type=float, help="Not implemented for DCGAN training")
    parser.add_argument(
        "--label_smooth", default=0, type=float, help="Discriminator label smoothing. Must be in [0, 0.5]."
    )

    loader_args, _ = parser.parse_known_args()
    fmaps = loader_args.fmaps
    dim = loader_args.dim
    label_noise = loader_args.label_noise
    label_smooth = loader_args.label_smooth

    if dim == 2 or dim == 3:
        raise NotImplementedError()
    elif dim == 4:
        dataloader = get_hmdb_loader(
            loader_args.batch_size,
            loader_args.num_workers,
            label_noise,
            label_smooth,
            preload=True,
            shape=(64, 64, 64),
        )

    parser = DCGAN.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    in_ch = 3 if dim == 4 else 1
    model = DCGAN(
        image_channels=in_ch,
        feature_maps_disc=fmaps,  # these need to be the same
        feature_maps_gen=fmaps,  # these need to be the same
        dimensionality=dim,
        latent_dim=1,
        label_noise=label_noise,
        label_smoothing=1 / label_smooth,
    )  # MNIST == 1 channel
    if dim == 2:
        sampler = TensorboardGenerativeModelImageSampler
    elif dim == 3:
        sampler = Tensorboard3dGenerativeModelImageSampler
    else:
        sampler = TensorboardVideoSampler
    callbacks = (
        [
            sampler(num_samples=10),
            LatentDimInterpolator(interpolate_epoch_interval=5),
        ]
        if dim < 4
        else [sampler(num_samples=10)]
    )
    trainer = Trainer.from_argparse_args(args, callbacks=callbacks, log_every_n_steps=10)
    trainer.fit(model, dataloader)
    model.manual_test(10)


if __name__ == "__main__":
    # Example run of this code on HMDB51 subset:
    # python main.py --gpus=1 --max_epochs=250 --dim=4 --latent_dim=4 --fmaps=32 --batch_size=4 --learning_rate=1e-5 --label_smooth=0.2
    test_train()
