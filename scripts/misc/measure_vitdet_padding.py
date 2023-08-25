#!/usr/bin/env python3

from pathlib import Path

from tqdm import tqdm

from datasets.vid import VIDResize, VID


def main():
    for size in 672, 1024:
        data = VID(
            Path("data", "vid"),
            split="vid_val",
            tar_path=Path("data", "vid", "data.tar"),
            combined_transform=VIDResize(
                short_edge_length=640 * size // 1024, max_size=size
            ),
        )
        weighted_sum = 0.0
        total_frames = 0
        # noinspection PyTypeChecker
        for vid_item in tqdm(data, ncols=0):
            frame = vid_item[0][0]
            padding_ratio = frame.shape[-1] * frame.shape[-2] / (size**2)
            weighted_sum += len(vid_item) * padding_ratio
            total_frames += len(vid_item)
        print(f"Size {size}: {weighted_sum / total_frames:.5g}")


if __name__ == "__main__":
    main()
