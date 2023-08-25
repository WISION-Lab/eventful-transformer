import json
import shutil
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from sys import stderr

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.io import read_image

from utils.image import rescale
from utils.misc import seeded_shuffle

CLASSES = [
    "airplane",
    "antelope",
    "bear",
    "bicycle",
    "bird",
    "bus",
    "car",
    "cattle",
    "dog",
    "domestic_cat",
    "elephant",
    "fox",
    "giant_panda",
    "hamster",
    "horse",
    "lion",
    "lizard",
    "monkey",
    "motorcycle",
    "rabbit",
    "red_panda",
    "sheep",
    "snake",
    "squirrel",
    "tiger",
    "train",
    "turtle",
    "watercraft",
    "whale",
    "zebra",
]

SPLITS = ["det_train", "vid_train", "vid_val", "vid_minival"]


class VID(Dataset):
    """
    A loader for the ImageNet VID dataset.
    """

    def __init__(
        self,
        location,
        split="vid_val",
        tar_path=None,
        shuffle=True,
        shuffle_seed=42,
        frame_transform=None,
        annotation_transform=None,
        combined_transform=None,
    ):
        """
        Initializes the loader. One the first call, this constructor
        will do some one-time setup.

        :param location: Directory containing the dataset (e.g.,
        data/vid). See the project README.
        :param split: Either "det_train", "vid_train", "vid_val", or
        "vid_minival"
        :param tar_path: Location of the data.tar file (e.g.,
        data/vid/data.tar). See the project README.
        :param shuffle: Whether to shuffle videos.
        :param shuffle_seed: The seed to use if shuffling.
        :param frame_transform: A callable to be applied to each frame
        as it is loaded. Passed to VIDItem constructor.
        :param annotation_transform: A callable to be applied to each
        bounding-box annotation as it is loaded. Passed to VIDItem
        constructor.
        :param combined_transform: A callable to be applied to each
        (frame, annotation) tuple as it is loaded. Passed to VIDItem
        constructor.
        """
        assert split in SPLITS
        self.frame_transform = frame_transform
        self.annotation_transform = annotation_transform
        self.combined_transform = combined_transform

        # Make sure the dataset has been set up.
        if not self.is_unpacked(location):
            assert tar_path is not None
            self.clean_unpacked(location)
            self.unpack(location, tar_path)

        # Load information about each video in the dataset.
        self.frames_path = Path(location, split, "frames")
        self.video_info = self._get_videos_info(location, split)

        # Optionally shuffle the videos (by default they are sorted).
        if shuffle:
            seeded_shuffle(self.video_info, shuffle_seed)

    def __getitem__(self, index):
        """
        Loads and returns an item from the dataset.

        :param index: The index of the item to load
        :return: A VIDItem object.
        """
        video_info = self.video_info[index]
        video_path = self.frames_path / video_info["video_id"]
        frame_paths = [
            str(video_path / frame["filename"]) for frame in video_info["frames"]
        ]
        annotations = [frame["annotations"] for frame in video_info["frames"]]
        vid_item = VIDItem(
            frame_paths,
            annotations,
            self.frame_transform,
            self.annotation_transform,
            self.combined_transform,
        )
        return vid_item

    def __len__(self):
        """
        Returns the number of items in the dataset.
        """
        return len(self.video_info)

    @staticmethod
    def clean_unpacked(location):
        """
        Deletes one-time setup data.

        :param location: The location of the dataset (see __init__)
        """
        base_path = Path(location)
        (base_path / "unpacked").unlink(missing_ok=True)
        for split in SPLITS:
            split_path = base_path / split
            if split_path.is_dir():
                shutil.rmtree(split_path)

    @staticmethod
    def is_unpacked(location):
        """
        Returns true if one-time setup has been completed.

        :param location: The location of the dataset (see __init__)
        """
        return Path(location, "unpacked").is_file()

    @staticmethod
    def unpack(location, tar_path):
        """
        Performs one-time setup. Extract data from the data.tar file.

        :param location: The location of the dataset (see __init__)
        :param tar_path: The location of the data.tar file (see
        __init__)
        """
        base_path = Path(location)
        base_path.mkdir(exist_ok=True)

        # Unpack the tar archive.
        print(f"Unpacking {tar_path.name}...", file=stderr, flush=True)
        shutil.unpack_archive(tar_path, base_path)
        unpacked_path = base_path / "vid_data"
        print("Unpacking complete.", file=stderr, flush=True)

        # Move the annotations.
        print(f"Reorganizing data...", file=stderr, flush=True)
        for split in SPLITS:
            split_path = base_path / split
            split_path.mkdir(exist_ok=True)
            annotations_path = unpacked_path / "annotations" / f"{split}.json"
            annotations_path.rename(split_path / "labels.json")

        # Reorganize the images.
        for split in SPLITS[:-1]:
            split_path = base_path / split
            frames_path = split_path / "frames"
            frames_path.mkdir(exist_ok=True)
            for filename in (unpacked_path / split).glob("*.JPEG"):
                video_id, frame_number = filename.stem.split("_")[-2:]
                video_path = frames_path / video_id
                video_path.mkdir(exist_ok=True)
                filename.rename(video_path / f"{frame_number}.jpg")

        # Symlink vid_minival/frames to vid_val/frames.
        link_from = base_path / SPLITS[-1] / "frames"
        link_to = base_path / SPLITS[-2] / "frames"
        link_from.symlink_to(link_to.resolve(), target_is_directory=True)
        print(f"Reorganization complete.", file=stderr, flush=True)

        # Clean up and create an empty indicator file.
        shutil.rmtree(unpacked_path)
        (base_path / "unpacked").touch()

    @staticmethod
    def _get_videos_info(location, split):
        with Path(location, split, "labels.json").open("r") as json_file:
            json_data = json.load(json_file)

        # Place frames in a dictionary with their ID as the key.
        frame_dict = {}
        for item in json_data["images"]:
            filename = Path(item["file_name"])
            video_id, frame_number = filename.stem.split("_")[-2:]
            frame_dict[item["id"]] = {
                "video_id": video_id,
                "filename": f"{frame_number}.jpg",
                "annotations": {"boxes": [], "labels": []},
            }

        # Assign each bounding box annotation to the correct frame.
        for item in json_data["annotations"]:
            annotations = frame_dict[item["image_id"]]["annotations"]
            # Convert from xywh to xyxy (what ViTDet outputs).
            x, y, w, h = item["bbox"]
            annotations["boxes"].append([x, y, x + w, y + h])

            # Convert to zero-based category IDs (what ViTDet outputs).
            annotations["labels"].append(item["category_id"] - 1)

        # Convert annotations to tensors and organize frames by video.
        video_dict = defaultdict(list)
        for frame in frame_dict.values():
            for key in "boxes", "labels":
                frame["annotations"][key] = torch.tensor(frame["annotations"][key])
            video_dict[frame.pop("video_id")].append(frame)
        videos_info = []
        for video_id, video in video_dict.items():
            video.sort(key=lambda v: v["filename"])
            # Some videos contain several non-contiguous segments. We
            # need to split these into distinct sequences.
            last = None
            segment = []
            for frame in video:
                i = int(Path(frame["filename"]).stem)
                if (last is not None) and (i > last + 1):
                    videos_info.append({"video_id": video_id, "frames": segment})
                    segment = []
                segment.append(frame)
                last = i
            if len(segment) > 0:
                videos_info.append({"video_id": video_id, "frames": segment})

        videos_info.sort(key=lambda v: v["video_id"] + v["frames"][0]["filename"])
        return videos_info


class VIDItem(Dataset):
    """
    A Dataset subclass for iterating over a single VID item. Necessary
    due to the very long length of some videos (loading into a single
    tensor would exhaust memory).
    """

    def __init__(
        self,
        frame_paths,
        annotations,
        frame_transform,
        annotation_transform,
        combined_transform,
    ):
        """
        Initializes the item.

        :param frame_paths: A list of frame paths for this item
        :param annotations: A list of annotations for this item
        :param frame_transform: A callable to be applied to each frame
        as it is loaded.
        :param annotation_transform: A callable to be applied to each
        bounding-box annotation as it is loaded.
        :param combined_transform: A callable to be applied to each
        (frame, annotation) tuple as it is loaded.
        """
        self.frame_paths = frame_paths
        self.annotations = annotations
        self.frame_transform = frame_transform
        self.annotation_transform = annotation_transform
        self.combined_transform = combined_transform

    def __getitem__(self, index):
        """
        Loads and returns a frame and the corresponding labels.

        :param index: The frame index
        :return: A (frame, annotations) tuple
        """
        frame = read_image(self.frame_paths[index])
        if self.frame_transform is not None:
            frame = self.frame_transform(frame)
        annotations = self.annotations[index]
        if self.annotation_transform is not None:
            annotations = self.annotation_transform(annotations)
        if self.combined_transform is not None:
            return self.combined_transform((frame, annotations))
        else:
            return frame, annotations

    def __len__(self):
        """
        Returns the number of frame in the item.
        """
        return len(self.frame_paths)


# short_edge_length=640 and max_size=1024:
# https://github.com/happyharrycn/detectron2_vitdet_vid/blob/main/projects/ViTDet-VID/configs/frcnn_vitdet.py#L103
class VIDResize(nn.Module):
    """
    A PyTorch module for simultaneously resizing frames and annotations.
    Should be passed to VID.__init__ as a combined_transform.
    """

    def __init__(self, short_edge_length, max_size):
        """

        :param short_edge_length: The size to which the short edge
        should be resized
        :param max_size: The maximum size of the long edge (this
        overrides short_edge_length if there is a conflict)
        """
        super().__init__()
        self.short_edge_length = short_edge_length
        self.max_size = max_size

    def forward(self, x):
        frame, annotations = x
        short_edge = min(frame.shape[-2:])
        long_edge = max(frame.shape[-2:])
        scale = min(self.short_edge_length / short_edge, self.max_size / long_edge)
        frame = rescale(frame, scale)
        annotations = deepcopy(annotations)
        annotations["boxes"] *= scale
        return frame, annotations
