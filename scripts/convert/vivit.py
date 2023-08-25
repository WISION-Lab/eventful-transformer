#!/usr/bin/env python3

from argparse import ArgumentParser

import torch

from utils.misc import parse_patterns, remap_weights


# Weight source:
# https://github.com/facebookresearch/detectron2/tree/main/projects/ViTDet


def main(args):
    in_weights = torch.load(args.in_file)["model_state"]
    patterns = parse_patterns(args.pattern_file)
    out_weights, n_remapped = remap_weights(in_weights, patterns, args.verbose)
    torch.save(out_weights, args.out_file)
    print(f"Remapped {n_remapped}/{len(in_weights)} weights.")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("in_file", help="the input .pyth file")
    parser.add_argument("out_file", help=".pth file where the output should be saved")
    parser.add_argument("pattern_file", help=".txt file containing regex patterns")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="print detailed output"
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
