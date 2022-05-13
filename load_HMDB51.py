from pathlib import Path
from typing import List

from joblib import Memory
from torch.utils.data import Dataset
from torchvision.datasets import HMDB51
from torchvision.io import write_video
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent
# Dataset downloaded from https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/#Downloads
DATA = ROOT / "data"
# annotations can be downloaded from
# http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/test_train_splits.rar
# which is listed in the torchvision.datasets.HMDB51 source code
ANNOTATIONS = ROOT / "annot/testTrainMulti_7030_splits"
SAMPLES = ROOT / "samples"

MEMOIZER = Memory(location=ROOT / "__JOBLIB__")

if not SAMPLES.exists():
    SAMPLES.mkdir(parents=True, exist_ok=True)


# Creating the dataset takes time. This caches a created dataset so loading
# is instant next time.
@MEMOIZER.cache()
def create_dataset(data: Path = DATA, annotations: Path = ANNOTATIONS) -> Dataset:
    return HMDB51(
        root=str(data), annotation_path=str(annotations), frame_rate=30, frames_per_clip=64, step_between_clips=128
    )


def get_dives_other(dataset: Dataset, other: str) -> List[int]:
    """Grabs samples from class "dive" and "other"."""
    idx = []
    dives, others = 0, 0
    for i in tqdm(range(len(dataset)), desc=f"Finding dives and {other}"):
        label_idx = dataset[i][-1]
        label = dataset.classes[label_idx]
        if label == "dive" and dives < 10:
            idx.append(i)
            dives += 1
        if label == other and others < 10:
            idx.append(i)
            others += 1
        if dives == 10:
            break
    return idx


if __name__ == "__main__":
    train = create_dataset()
    print(train.classes)
    # line below gets the indices of some "diving" and "ride_horse" samples
    # indices = get_dives_other(train, "walk")
    # print(indices, len(indices))
    indices = [1752, 1753, 1754, 1755, 1756, 474, 476, 478, 480, 482]  # 10

    for idx in indices:
        video, audio, label_idx = train[idx]
        label = train.classes[label_idx]
        try:
            path = train.video_clips_metadata["video_paths"][idx]
        except:
            path = "???"

        print("\nSize of first video (sample):")
        print(f"         Path: {path}")
        print(f"  Video shape: {video.shape} [format = (T, H, W, C)]")
        print(f"  Audio shape: {audio.shape}")
        print(f"  Class label: {label}")
        outfile = str(SAMPLES / f"vid_sample{idx}.mkv")
        write_video(outfile, video, fps=30, video_codec="h264")
        print(f"  Sample written to {outfile}.")
