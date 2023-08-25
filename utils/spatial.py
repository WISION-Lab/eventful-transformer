from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from eventful_transformer.policies import TokenNormTopK
from models.vivit import FactorizedViViT
from utils.misc import get_pytorch_device, set_policies


def compute_vivit_spatial(config, output_dir, data):
    device = get_pytorch_device()

    # Load and set up the model.
    model = FactorizedViViT(**(config["model"]))
    model.load_state_dict(torch.load(config["weights"]))
    model = model.to(device)

    set_policies(model, TokenNormTopK, k=config["k"])
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    data = DataLoader(data, batch_size=1, drop_last=False)
    for i, (video, label) in tqdm(enumerate(data), total=len(data), ncols=0):
        model.reset()
        with torch.inference_mode():
            spatial = model(video.to(device))
            np.savez(
                output_dir / f"{i:05d}.npz",
                spatial=spatial.cpu().numpy(),
                label=label[0].cpu().numpy(),
            )
