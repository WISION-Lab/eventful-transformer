#!/usr/bin/env python3

from pathlib import Path

from datasets.kinetics400 import Kinetics400
from utils.config import get_cli_config
from utils.spatial import compute_vivit_spatial


def main():
    config = get_cli_config(
        config_location=Path("configs", "spatial", "vivit_kinetics400")
    )
    k = config["k"]
    location = Path("data", "kinetics400")
    for split in "train", "val":
        print(f"{split.capitalize()}, k={k}", flush=True)
        max_tars = config.get("max_tars", None) if (split == "train") else None
        data = Kinetics400(
            location,
            split=split,
            decode_size=224,
            decode_fps=25,
            max_tars=max_tars,
            shuffle=False,
        )
        if max_tars is not None:
            split = f"{split}_{max_tars}"
        compute_vivit_spatial(config, location / split / f"spatial_224_25_{k}", data)


if __name__ == "__main__":
    main()
