import os
import sys
import time
import torch
import shutil

from utilities import *
from datasets import FigaroDataset
from networks import EfficientVitUNet
from accelerate import Accelerator
from trainer import evaluate_model, train_model




class helper:

    train_loader = None
    test_loader = None

    @staticmethod
    def get_config():
        config = {
            # common
            "dataset_folder": "datasets/Figaro-1k",
            "time_bar": time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()),
            # dataloader
            "batch_size": 64,
            "num_workers": 12,
            # optimizer
            "epoch": 150,
            "lr": 2e-3,
        }
        config["result_folder"] = os.path.join(
            "results", f"Figaro-{config['time_bar']}"
        )
        return config

    @staticmethod
    def get_train_loader(config):
        if helper.train_loader is None:
            helper.train_loader = torch.utils.data.DataLoader(
                FigaroDataset(config["dataset_folder"], "train"),
                batch_size=config["batch_size"],
                num_workers=config["num_workers"],
                persistent_workers=False,
                pin_memory=False,
            )
        return helper.train_loader

    @staticmethod
    def get_test_loader(config):
        if helper.test_loader is None:
            helper.test_loader = torch.utils.data.DataLoader(
                FigaroDataset(config["dataset_folder"], "test"),
                batch_size=config["batch_size"],
                num_workers=config["num_workers"],
                persistent_workers=False,
                pin_memory=False,
            )
        return helper.test_loader

    @staticmethod
    def get_model():
        return EfficientVitUNet(
            layers=[5, 5, 15, 10],
            embed_dims=[40, 80, 192, 384],
            downsamples=[True, True, True, True],
            resolution=256,
            num_classes=1,
            in_channels=3,
        )


def run_experiment(config, entry):
    os.makedirs(config["result_folder"])
    try:
        log_path = os.path.join(config["result_folder"], "log.txt")
        with open(log_path, "w") as log_file:
            sys.stdout = StandardOutputDuplicator(log_file)

            results = entry(config)

            write_json_result(
                results, os.path.join(config["result_folder"], "results.json")
            )
            write_json_result(
                config, os.path.join(config["result_folder"], "config.json")
            )
    except Exception as e:
        shutil.rmtree(config["result_folder"])
        raise e


def train(config):
    accelerator = Accelerator()
    model = helper.get_model()

    train_model(config, model, accelerator, helper, verbose=True)

    test_loss, test_dice = evaluate_model(
        config, model, accelerator, helper, verbose=True
    )

    torch.save(model.state_dict(), os.path.join(config["result_folder"], "model.pth"))

    return {
        "test_loss": test_loss,
        "test_dice": test_dice,
    }


def main():
    config = helper.get_config()
    run_experiment(config, train)


if __name__ == "__main__":
    main()
