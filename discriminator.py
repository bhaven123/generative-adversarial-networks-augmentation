# Implementation adapted from
# https://github.com/PyTorchLightning/lightning-bolts/blob/8b4d90d7443ac263fd246e2745c267439f4b9274/pl_bolts/models/gans/dcgan/components.py#L6
from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from torch import Tensor
from torch.nn import BatchNorm2d, BatchNorm3d, Conv2d, Conv3d, LeakyReLU, Module, Sequential, Sigmoid

FloatArray = NDArray[np.floating]


class Discriminator2d(Module):
    def __init__(self, in_channels: int, n_feature_maps: int):
        super().__init__()
        f = int(n_feature_maps)
        self.model = Sequential(
            Conv2d(in_channels, n_feature_maps, kernel_size=4, stride=2, padding=1, bias=True),
            LeakyReLU(0.2, inplace=True),
            *self._conv_group(in_channels=f, out_channels=f * 2),
            *self._conv_group(in_channels=f * 2, out_channels=f * 4),
            *self._conv_group(in_channels=f * 4, out_channels=f * 8),
            Conv2d(in_channels=f * 8, out_channels=1, kernel_size=4, stride=1, padding=0, bias=True),
            Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x).view(-1, 1).squeeze(1)

    @staticmethod
    def _conv_group(
        in_channels: int, out_channels: int, kernel_size: int = 4, stride: int = 2, padding: int = 1
    ) -> Tuple[Module, Module, Module]:
        k, s, p = kernel_size, stride, padding
        return (
            Conv2d(in_channels, out_channels, kernel_size=k, stride=s, padding=p, bias=False),
            BatchNorm2d(out_channels),
            LeakyReLU(0.2, inplace=True),
        )


class Discriminator3d(Module):
    def __init__(self, in_channels: int, n_feature_maps: int):
        super().__init__()
        f = int(n_feature_maps)
        self.model = Sequential(
            Conv3d(in_channels, n_feature_maps, kernel_size=4, stride=2, padding=1, bias=True),
            LeakyReLU(0.2, inplace=True),
            *self._conv_group(in_channels=f, out_channels=f * 2),
            *self._conv_group(in_channels=f * 2, out_channels=f * 4),
            *self._conv_group(in_channels=f * 4, out_channels=f * 8),
            Conv3d(in_channels=f * 8, out_channels=1, kernel_size=4, stride=1, padding=0, bias=True),
            Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x).view(-1, 1).squeeze(1)

    @staticmethod
    def _conv_group(
        in_channels: int, out_channels: int, kernel_size: int = 4, stride: int = 2, padding: int = 1
    ) -> Tuple[Module, Module, Module]:
        k, s, p = kernel_size, stride, padding
        return (
            Conv3d(in_channels, out_channels, kernel_size=k, stride=s, padding=p, bias=False),
            BatchNorm3d(out_channels),
            LeakyReLU(0.2, inplace=True),
        )


# Tried to implement R(2+1)D Network
# class Discriminator2Plus1D(Module):
#     def __init__(self, in_channels: int, mid_channels: int, n_feature_maps: int):
#         super().__init__()
#         mid_channels = (in_channels * planes * 3 * 3 * 3) // (in_channels * 3 * 3 + 3 * planes)
#         f = int(n_feature_maps)
#         self.model = Sequential(
#             Conv3d(in_channels, n_feature_maps, kernel_size=4, stride=2, padding=1, bias=True),
#             LeakyReLU(0.2, inplace=True),
#             *self._conv_group(in_channels=f, out_channels=f * 2),
#             *self._conv_group(in_channels=f * 2, out_channels=f * 4),
#             *self._conv_group(in_channels=f * 4, out_channels=f * 8),
#             Conv3d(in_channels=f * 8, out_channels=1, kernel_size=4, stride=1, padding=0, bias=True),
#             Sigmoid(),
#         )

#     def forward(self, x: Tensor) -> Tensor:
#         return self.model(x).view(-1, 1).squeeze(1)

#     @staticmethod
#     def _conv_group(
#         in_channels: int, mid_channels: int, out_channels: int, kernel_size: int = 4, stride: int = 2, padding: int = 1
#     ) -> Tuple[Module, Module, Module]:
#         k, s, p = kernel_size, stride, padding
#         return (
#             Conv3d(
#                 in_channels,
#                 mid_channels,
#                 kernel_size=(1, 3, 3),
#                 stride=(1, stride, stride),
#                 padding=(0, padding, padding),
#                 bias=False,
#             ),
#             BatchNorm3d(out_channels),
#             LeakyReLU(0.2, inplace=True),
#             Conv3d(
#                 mid_channels,
#                 out_channels,
#                 kernel_size=(3, 1, 1),
#                 stride=(stride, 1, 1),
#                 padding=(padding, 0, 0),
#                 bias=False,
#             ),
#         )
