#!/usr/bin/env python3

from pathlib import Path

from datasets.vivit_spatial import ViViTSpatial
from utils.config import get_cli_config
from utils.train import train_vivit_temporal


def main():
    config = get_cli_config(
        config_location=Path("configs", "train", "vivit_kinetics400")
    )
    train_data = ViViTSpatial(
        Path("data", "kinetics400"),
        split="train_40",
        base_name="spatial_224_25",
        k=config["k"],
    )
    val_data = ViViTSpatial(
        Path("data", "kinetics400"),
        split="val",
        base_name="spatial_224_25",
        k=config["k"],
    )
    train_vivit_temporal(config, train_data, val_data)


if __name__ == "__main__":
    main()
