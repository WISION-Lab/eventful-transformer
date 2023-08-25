#!/usr/bin/env python3

import pickle
from argparse import ArgumentParser

import torch

from utils.misc import parse_patterns, remap_weights


# Weight sources:
# https://github.com/alibaba-mmai-research/TAdaConv/blob/main/MODEL_ZOO.md
# https://github.com/happyharrycn/detectron2_vitdet_vid/tree/main/projects/ViTDet-VID


def main(args):
    if args.in_file.endswith(".pkl"):
        with open(args.in_file, "rb") as pickle_file:
            in_weights = pickle.load(pickle_file)
    else:
        in_weights = torch.load(args.in_file)
    in_weights = in_weights["model"]

    # Throw out the class embedding token.
    in_weights["backbone.net.pos_embed"] = in_weights["backbone.net.pos_embed"][:, 1:]

    patterns = parse_patterns(args.pattern_file)
    out_weights, n_remapped = remap_weights(in_weights, patterns, args.verbose)
    for key, weight in out_weights.items():
        # Modifying in place while iterating is okay because the keys
        # aren't changing.
        if not isinstance(out_weights[key], torch.Tensor):
            out_weights[key] = torch.tensor(weight)
    torch.save(out_weights, args.out_file)
    print(f"Remapped {n_remapped}/{len(in_weights)} weights.")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("in_file", help="the input .pkl or .pth file")
    parser.add_argument("out_file", help=".pth file where the output should be saved")
    parser.add_argument("pattern_file", help=".txt file containing regex patterns")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="print detailed output"
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
