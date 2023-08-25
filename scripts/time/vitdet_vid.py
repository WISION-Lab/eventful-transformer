#!/usr/bin/env python3

import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.vid import VIDResize, VID
from models.vitdet import ViTDet
from utils.config import initialize_run
from utils.evaluate import run_evaluations
from utils.misc import MeanValue


def evaluate_vitdet_runtime(device, model, data, config):
    model.no_counting()
    backbone = MeanValue()
    backbone_non_first = MeanValue()
    other = MeanValue()
    other_non_first = MeanValue()
    n_items = config.get("n_items", len(data))
    for _, vid_item in tqdm(zip(range(n_items), data), total=n_items, ncols=0):
        vid_item = DataLoader(vid_item, batch_size=1)
        model.reset()
        for t, (frame, annotations) in enumerate(vid_item):
            with torch.inference_mode():
                frame = frame.to(device)
                torch.cuda.synchronize()
                t_0 = time.time()
                images, x = model.pre_backbone(frame)
                torch.cuda.synchronize()
                t_1 = time.time()
                x = model.backbone(x)
                torch.cuda.synchronize()
                t_2 = time.time()
                model.post_backbone(images, x)
                torch.cuda.synchronize()
                t_3 = time.time()
                t_backbone = t_2 - t_1
                t_other = (t_3 - t_2) + (t_1 - t_0)
                backbone.update(t_backbone)
                other.update(t_other)
                if t > 0:
                    backbone_non_first.update(t_backbone)
                    other_non_first.update(t_other)
    times = {
        "backbone": backbone.compute(),
        "backbone_non_first": backbone_non_first.compute(),
        "other": other.compute(),
        "other_non_first": other_non_first.compute(),
        "total": backbone.compute() + other.compute(),
        "total_non_first": backbone_non_first.compute() + other_non_first.compute(),
    }
    return {"times": times}


def main():
    config = initialize_run(config_location=Path("configs", "time", "vitdet_vid"))
    input_size = config.get("input_size", 1024)
    data = VID(
        Path("data", "vid"),
        split=config["split"],
        tar_path=Path("data", "vid", "data.tar"),
        combined_transform=VIDResize(
            short_edge_length=640 * input_size // 1024, max_size=input_size
        ),
    )
    run_evaluations(config, ViTDet, data, evaluate_vitdet_runtime)


if __name__ == "__main__":
    main()
