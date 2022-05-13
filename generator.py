# Implementation adapted from
# https://github.com/PyTorchLightning/lightning-bolts/blob/8b4d90d7443ac263fd246e2745c267439f4b9274/pl_bolts/models/gans/dcgan/components.py#L6
from typing import Tuple

from torch import Tensor
from torch.nn import BatchNorm2d, BatchNorm3d, ConvTranspose2d, ConvTranspose3d, LeakyReLU, Module, Sequential, Tanh


class Generator2d(Module):
    def __init__(self, in_channels: int, n_feature_maps: int, out_channels: int):
        super().__init__()
        f = int(n_feature_maps)
        self.model = Sequential(
            *self._conv_group(in_channels=in_channels, out_channels=f * 8, kernel_size=4, stride=1, padding=0),
            *self._conv_group(in_channels=f * 8, out_channels=f * 4),
            *self._conv_group(in_channels=f * 4, out_channels=f * 2),
            *self._conv_group(in_channels=f * 2, out_channels=f),
            ConvTranspose2d(in_channels=f, out_channels=out_channels, kernel_size=4, stride=2, padding=1, bias=True),
            Tanh(),
        )

    def forward(self, z: Tensor) -> Tensor:
        return self.model(z)

    @staticmethod
    def _conv_group(
        in_channels: int, out_channels: int, kernel_size: int = 4, stride: int = 2, padding: int = 1
    ) -> Tuple[Module, Module, Module]:
        k, s, p = kernel_size, stride, padding
        return (
            ConvTranspose2d(in_channels, out_channels, kernel_size=k, stride=s, padding=p, bias=False),
            BatchNorm2d(out_channels),
            LeakyReLU(inplace=True),
        )


class Generator3d(Module):
    def __init__(self, in_channels: int, n_feature_maps: int, out_channels: int):
        super().__init__()
        f = int(n_feature_maps)
        self.model = Sequential(
            *self._conv_group(in_channels=in_channels, out_channels=f * 8, kernel_size=4, stride=1, padding=0),
            *self._conv_group(in_channels=f * 8, out_channels=f * 4),
            *self._conv_group(in_channels=f * 4, out_channels=f * 2),
            *self._conv_group(in_channels=f * 2, out_channels=f),
            ConvTranspose3d(in_channels=f, out_channels=out_channels, kernel_size=4, stride=2, padding=1, bias=True),
            Tanh(),
        )

    def forward(self, z: Tensor) -> Tensor:
        return self.model(z)

    @staticmethod
    def _conv_group(
        in_channels: int, out_channels: int, kernel_size: int = 4, stride: int = 2, padding: int = 1
    ) -> Tuple[Module, Module, Module]:
        k, s, p = kernel_size, stride, padding
        return (
            ConvTranspose3d(in_channels, out_channels, kernel_size=k, stride=s, padding=p, bias=False),
            BatchNorm3d(out_channels),
            LeakyReLU(inplace=True),
        )
