from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.misc import seeded_shuffle


class ViViTSpatial(Dataset):
    """
    A loader for intermediate outputs of the ViViT spatial model. See
    scripts/train/vivit_epic_kitchens.py.
    """

    def __init__(
        self,
        location,
        split="train",
        base_name="spatial",
        k=None,
        shuffle=True,
        shuffle_seed=42,
    ):
        """
        Initializes the loader.

        :param location: Location where the base dataset is stored
        (e.g., data/epic_kitchens)
        :param split: The split for the base dataset (e.g., "train")
        :param base_name: The name of the intermediate output folder
        containing .npz files (e.g., "spatial_50")
        :param k: If not None, this is appended as f"{base_name}_{k}"
        :param shuffle: Whether to shuffle items
        :param shuffle_seed: The seed to use if shuffling
        """
        # Load the path of each item in the dataset.
        name = base_name if (k is None) else f"{base_name}_{k}"
        paths = sorted(Path(location, split, name).glob("*.npz"))
        self.item_paths = [str(path) for path in paths]

        # Optionally shuffle the items.
        if shuffle:
            seeded_shuffle(self.item_paths, shuffle_seed)

    def __getitem__(self, index):
        """
        Loads and returns an item from the dataset.

        :param index: The index of the item to load
        :return: A (spatial, label) tuple, where "spatial" is the
        intermediate output of the ViViT spatial model, and "label" is
        the class label.
        """
        item = np.load(self.item_paths[index])
        return torch.tensor(item["spatial"]), torch.tensor(item["label"])

    def __len__(self):
        """
        Returns the number of items in the dataset.
        """
        return len(self.item_paths)
