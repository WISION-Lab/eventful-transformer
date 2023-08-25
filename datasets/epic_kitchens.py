import csv
import shutil
from pathlib import Path
from sys import stderr

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from tqdm import tqdm

from utils.misc import decode_video, seeded_shuffle

SPLITS = ["train", "validation"]


class EPICKitchens(Dataset):
    """
    A loader for the EPIC-Kitchens 100 dataset.
    """

    def __init__(
        self,
        location,
        split="validation",
        shuffle=True,
        shuffle_seed=42,
        video_transform=None,
    ):
        """
        Initializes the loader. On the first call, this constructor will
        do some one-time setup.

        :param location: Directory containing the dataset (e.g.,
        data/epic_kitchens). See the project README.
        :param split: Either "train" or "validation"
        :param shuffle: Whether to shuffle videos
        :param shuffle_seed: The seed to use if shuffling
        :param video_transform: A callable to be applied to each video
        as it is loaded
        """
        assert split in SPLITS
        self.video_transform = video_transform

        # Make sure the dataset has been set up.
        Path(location, split).mkdir(parents=True, exist_ok=True)
        if not self.is_decoded(location, split):
            self.clean_decoded(location, split)
            self.decode(location, split)

        # Load information about each clip in the dataset.
        self.frames_path = Path(location, split, "frames")
        self.clips_info = self._get_clips_info(location, split)

        # Optionally shuffle the videos.
        if shuffle:
            seeded_shuffle(self.clips_info, shuffle_seed)

    def __getitem__(self, index):
        """
        Loads and returns an item from the dataset.

        :param index: The index of the item to load
        :return: A (video, info) tuple, where "video" is a tensor and
        "info" is a dict containing the label and other metadata
        """
        clip_info = self.clips_info[index]
        clip_path = self.frames_path / f"{clip_info['clip_id']:05d}"
        frame_paths = sorted(clip_path.glob("*.jpg"))
        video = torch.stack([read_image(str(frame_path)) for frame_path in frame_paths])
        if self.video_transform is not None:
            video = self.video_transform(video)
        return video, clip_info["class_id"]

    def __len__(self):
        """
        Returns the number of items in the dataset.
        """
        return len(self.clips_info)

    @staticmethod
    def clean_decoded(location, split):
        """
        Deletes one-time setup data.

        :param location: The location of the dataset (see __init__)
        :param split: The split ("train" or "validation")
        """
        base_path = Path(location, split)
        (base_path / "decoded").unlink(missing_ok=True)
        folder_path = base_path / "frames"
        if folder_path.is_dir():
            shutil.rmtree(folder_path)

    @staticmethod
    def decode(location, split):
        """
        Performs one-time setup. Decode/extract videos based on
        information in the CSV files.

        :param location: The location of the dataset (see __init__)
        :param split: The split ("train" or "validation")
        """
        base_path = Path(location, split)
        frames_path = base_path / "frames"
        frames_path.mkdir(exist_ok=True)

        # Decode videos into images.
        print("Decoding clips...", file=stderr, flush=True)
        clips_info = EPICKitchens._get_clips_info(location, split)
        for clip_info in tqdm(clips_info, total=len(clips_info), ncols=0):
            video_path = Path(location, "videos", f"{clip_info['video_id']}.mp4")
            decode_path = frames_path / f"{clip_info['clip_id']:05d}"
            ffmpeg_input_args = [
                "-ss",
                clip_info["start_time"],
                "-to",
                clip_info["end_time"],
            ]
            ffmpeg_output_args = ["-qscale:v", "2"]
            return_code = decode_video(
                video_path,
                decode_path,
                name_format="%4d",
                image_format="jpg",
                ffmpeg_input_args=ffmpeg_input_args,
                ffmpeg_output_args=ffmpeg_output_args,
            )
            if return_code != 0:
                print(
                    f"Decoding failed for clip {clip_info['clip_id']}",
                    file=stderr,
                    flush=True,
                )
                shutil.rmtree(decode_path)

        # Create an empty indicator file.
        print("Decoding complete.", file=stderr, flush=True)
        (base_path / f"decoded").touch()

    @staticmethod
    def is_decoded(location, split):
        """
        Returns true if one-time setup has been completed.

        :param location: The location of the dataset (see __init__)
        :param split: The split ("train" or "validation")
        """
        return Path(location, split, "decoded").is_file()

    @staticmethod
    def _get_clips_info(location, split):
        clips_info = []
        with open(Path(location, f"EPIC_100_{split}.csv"), "r") as csv_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader)  # Skip header line
            for i, line in enumerate(csv_reader):
                clips_info.append(
                    {
                        "clip_id": i,
                        "video_id": line[2],
                        "start_time": line[4],
                        "end_time": line[5],
                        "label": line[9],
                        "class_id": int(line[10]),
                    }
                )
        return clips_info
