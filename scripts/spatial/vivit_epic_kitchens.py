#!/usr/bin/env python3

from pathlib import Path

from datasets.epic_kitchens import EPICKitchens
from utils.config import get_cli_config
from utils.spatial import compute_vivit_spatial


def main():
    config = get_cli_config(
        config_location=Path("configs", "spatial", "vivit_epic_kitchens")
    )
    k = config["k"]
    location = Path("data", "epic_kitchens")
    for split in "train", "validation":
        print(f"{split.capitalize()}, k={k}", flush=True)
        data = EPICKitchens(location, split=split, shuffle=False)
        compute_vivit_spatial(config, location / split / f"spatial_{k}", data)


if __name__ == "__main__":
    main()
