# Implementation adapted from https://lightning-flash.readthedocs.io/en/latest/quickstart.html
import torch

import flash
from flash.video import VideoClassificationData, VideoClassifier
from torch.utils.data import Dataset
from pytorch_lightning import seed_everything
from pathlib import Path
from joblib import Memory
import torchmetrics
from argparse import ArgumentParser
import warnings


ROOT = Path(__file__).resolve().parent
TRAIN_DATA = Path(__file__).resolve().parent / "data"
BATCH_SIZE = 8
NUM_WORKERS = 8

MEMOIZER = Memory(location=ROOT / "__JOBLIB__")

warnings.filterwarnings("ignore")

# Get the backbones available for VideoClassifier
backbones = VideoClassifier.available_backbones()

# Print the backbones
print("Availabe pretrained video models:", backbones)


@MEMOIZER.cache()
def train_data(data: Path = TRAIN_DATA) -> Dataset:
    return VideoClassificationData.from_folders(
        train_folder=data,
        clip_sampler="uniform",
        clip_duration=2,
        decode_audio=False,
        batch_size=4,
    )


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--backbone", default="x3d_xs", type=str, help="Video Models available in PyTorchVideo.")
    parser.add_argument(
        "--pretrained",
        default="True",
        choices=["True", "False"],
        type=str,
        help="Whether the model is pretrained or not.",
    )
    parser.add_argument(
        "--task",
        default="finetune",
        choices=["finetune", "train"],
        type=str,
        help="To finetune a pretrained model on custom data or train if more data is available.",
    )
    loader_args, _ = parser.parse_known_args()
    backbone = loader_args.backbone
    pretrained = loader_args.pretrained
    option = loader_args.task
    seed_everything(42)
    torch.cuda.empty_cache()

    # 1. Create the DataModule
    datamodule = train_data()

    # 2. Build the task
    metrics_ = [torchmetrics.Accuracy(), torchmetrics.F1(num_classes=datamodule.num_classes)]
    model = VideoClassifier(
        backbone=backbone,
        labels=datamodule.labels,
        num_classes=datamodule.num_classes,
        pretrained=pretrained,
        metrics=metrics_,
    )

    # 3. Create the trainer
    trainer = flash.Trainer(
        max_epochs=5,
        gpus=torch.cuda.device_count(),
        strategy="ddp" if torch.cuda.device_count() > 1 else None,
    )
    # Finetune Pretrained model
    if option == "finetune":
        trainer.finetune(model, datamodule=datamodule, strategy="freeze")
    # Retrain the model
    elif option == "train":
        trainer.fit(model, datamodule=datamodule)

    # 4. Make a prediction
    datamodule_pred = VideoClassificationData.from_folders(predict_folder="predict", batch_size=1)
    predictions = trainer.predict(model, datamodule=datamodule_pred, output="labels")
    print(predictions)

    # 5. Save the model!
    trainer.save_checkpoint("video_classification.pt")


if __name__ == "__main__":
    # python flash_video_classifier.py --backbone x3d_xs --pretrained True --task finetune
    # The working backbones are x3d_xs, c2d_r50, and i3d_r50. Tested on x3d_xs.
    main()
