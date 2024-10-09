import sys
sys.path.append("src")

import torch
torch.set_float32_matmul_precision("medium")

from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers import CSVLogger

from face_classify.trainer import Trainer
from face_classify.data.datamodule import DataModule

if __name__ == "__main__":
    cli = LightningCLI(
        model_class=Trainer,
        seed_everything_default=42,
        datamodule_class=DataModule,
        trainer_defaults={
            "logger": {
                "class_path": "CSVLogger",
                "init_args": {
                    "save_dir": "logs",
                    "name": "csv_logger"
                },
            },
            "max_epochs": 10,
            "accelerator": "cuda"
        }   
    )