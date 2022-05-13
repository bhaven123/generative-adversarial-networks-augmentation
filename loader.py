from pathlib import Path
from typing import Callable, List, Tuple
from warnings import filterwarnings

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.distributions.uniform import Uniform
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from tqdm import tqdm

from load_HMDB51 import create_dataset

FloatArray = NDArray[np.floating]

ROOT = Path(__file__).resolve().parent
IMAGE_SIZE = 64
TRANSFORMS = Compose(
    [
        Resize(IMAGE_SIZE),
        ToTensor(),
        Normalize((0.5,), (0.5,)),
    ]
)
VID_TRANSFORM = Compose(
    [
        Normalize((0.6620, 0.6620, 0.6620), (0.1161, 0.1270, 0.1249)),  # means and sds of channels of second video
    ]
)


def normalize_video(shape: Tuple[int, int, int] = (128, 240, 320)) -> Tensor:
    def resize(vid: Tensor) -> Tensor:
        vid = torch.true_divide(vid, 255)
        vid = torch.nn.functional.interpolate(vid.unsqueeze(0), shape, mode="trilinear", align_corners=False).squeeze(0)
        return vid

    return resize


class HMDBDataset(Dataset):
    """A thin wrapper around the torchvision.io hmdb51 loader.

    hmdb_data: Dataset,
        The torchvision loader, ready and initialized.

    label_noise: float = 0.1,
        ONLY RELEVANT IF BUILDING A CONDITIONAL GAN. Determines how often a label
        should be given a random valid label in the dataset.

    label_smoothing: float = 0.3,
        ONLY RELEVANT IF BUILDING A CONDITIONAL GAN. Determines how much uniform
        noise to add to a label. Must be in [0, 0.5) to make sense.

    transform: Callable = normalize_video(),
        And transform to use. The default resizes videos to (64, 64, 64), which
        is necessary for DCGAN to work and converge, and also normalizes the
        data by dividing by 255 to rescale the uint8 values.
    """

    def __init__(
        self,
        hmdb_data: Dataset,
        label_noise: float = 0.1,
        label_smoothing: float = 0.3,
        transform: Callable = normalize_video(),
    ) -> None:
        self.label_noise = label_noise
        self.label_smoothing = label_smoothing
        self.unif = Uniform(-label_smoothing / 2, label_smoothing / 2)
        self.transform = transform
        self.label_values = list(range(len(hmdb_data.classes)))

        self.data = hmdb_data
        self.label_values = list(range(len(self.data.classes)))

    def __getitem__(self, i: int) -> Tuple[Tensor, Tensor]:
        filterwarnings("ignore", "The pts_unit 'pts'", category=UserWarning)
        x, _, y = self.data[i]  # x.shape == (T, H, W, C)
        if self.label_smoothing > 0:
            y += self.unif.sample()
        if self.label_noise > 0:
            if torch.rand([1]) < self.label_noise:
                y = torch.randint_like(y, 0, len(self.label_values))
        x = x.permute([3, 0, 1, 2])  # x.shape == (C, T, H, W)
        x = self.transform(x)
        return x, y

    def __len__(self) -> int:
        return len(self.data)


class PreloadedHMDBDataset(HMDBDataset):
    """Wrapper around the dataset above that loads the data in `indices` and holds all those clips in memory
    to greatly accelerate loading and training. Otherwise, the default would be to read an entire video from
    disk and extract clips each time...
    """

    def __init__(
        self,
        hmdb_data: Dataset,
        label_noise: float = 0.1,
        label_smoothing: float = 0.3,
        transform: Callable = normalize_video(),
        indices: List[int] = [],
    ) -> None:
        filterwarnings("ignore", "The pts_unit 'pts'", category=UserWarning)
        super().__init__(hmdb_data, label_noise=label_noise, label_smoothing=label_smoothing, transform=transform)
        dataset = Subset(self.data, indices)

        data = []
        for i in tqdm(range(len(dataset)), desc="Preloading data"):
            x, _, y = dataset[i]  # x.shape == (T, H, W, C)
            if self.label_smoothing > 0:
                y += self.unif.sample()
            if self.label_noise > 0:
                if torch.rand([1]) < self.label_noise:
                    y = torch.randint_like(y, 0, len(self.label_values))
            x = x.permute([3, 0, 1, 2])  # x.shape == (C, T, H, W)
            x = self.transform(x)
            data.append((x, y))
        self.data = data

    def __getitem__(self, i: int) -> Tuple[Tensor, Tensor]:
        return self.data[i]


def get_hmdb_loader(
    batch_size: int = 8,
    num_workers: int = 8,
    label_noise: float = 0,
    label_smooth: float = 0,
    shape: Tuple[int, int, int] = (128, 240, 320),
    preload: bool = False,
) -> DataLoader:
    indices = [1752, 1753, 1754, 1755, 1756, 474, 476, 478, 480, 482]  # 10
    data = create_dataset()
    args = dict(hmdb_data=data, label_noise=label_noise, label_smoothing=label_smooth, transform=normalize_video(shape))
    dataset = HMDBDataset(**args) if not preload else PreloadedHMDBDataset(indices=indices, **args)
    # dataset = Subset(dataset, np.arange(64))
    if not preload:
        dataset = Subset(dataset, indices)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return dataloader


if __name__ == "__main__":
    print("Data Loader")
