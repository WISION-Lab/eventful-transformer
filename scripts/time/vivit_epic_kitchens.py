#!/usr/bin/env python3

import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.epic_kitchens import EPICKitchens
from models.vivit import FactorizedViViT
from utils.config import initialize_run
from utils.evaluate import run_evaluations
from utils.misc import MeanValue


def evaluate_vivit_runtime(device, model, data, config):
    model.no_counting()
    spatial = MeanValue()
    temporal = MeanValue()
    data = DataLoader(data, batch_size=1)
    n_items = config.get("n_items", len(data))
    for _, (video, label) in tqdm(zip(range(n_items), data), total=n_items, ncols=0):
        model.reset()
        with torch.inference_mode():
            video = video.to(device)
            torch.cuda.synchronize()
            t_0 = time.time()
            model.spatial_only = True
            model.temporal_only = False
            x = model(video)
            torch.cuda.synchronize()
            t_1 = time.time()
            model.spatial_only = False
            model.temporal_only = True
            model(x)
            t_2 = time.time()
            torch.cuda.synchronize()
        spatial.update(t_1 - t_0)
        temporal.update(t_2 - t_1)
    times = {
        "spatial": spatial.compute(),
        "temporal": temporal.compute(),
        "total": spatial.compute() + temporal.compute(),
    }
    return {"times": times}


def main():
    config = initialize_run(
        config_location=Path("configs", "time", "vivit_epic_kitchens")
    )
    data = EPICKitchens(Path("data", "epic_kitchens"), split="validation")
    run_evaluations(config, FactorizedViViT, data, evaluate_vivit_runtime)


if __name__ == "__main__":
    main()
