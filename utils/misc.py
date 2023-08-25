import re
import subprocess
from pathlib import Path
from random import Random

import requests
import torch

from eventful_transformer.modules import SimpleSTGTGate, TokenDeltaGate, TokenGate


class MeanValue:
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def compute(self):
        return 0.0 if (self.count == 0) else self.sum / self.count

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, value):
        self.sum += value
        self.count += 1


class TopKAccuracy:
    def __init__(self, k):
        self.k = k
        self.correct = 0
        self.total = 0

    def compute(self):
        return self.correct / self.total

    def reset(self):
        self.correct = 0
        self.total = 0

    def update(self, pred, true):
        _, top_k = pred.topk(self.k, dim=-1)
        self.correct += true.eq(top_k).sum().item()
        self.total += true.numel()


def decode_video(
    input_path,
    output_path,
    name_format="%d",
    image_format="png",
    ffmpeg_input_args=None,
    ffmpeg_output_args=None,
):
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True)
    if ffmpeg_input_args is None:
        ffmpeg_input_args = []
    if ffmpeg_output_args is None:
        ffmpeg_output_args = []
    return subprocess.call(
        ["ffmpeg", "-loglevel", "error"]
        + ffmpeg_input_args
        + ["-i", input_path]
        + ffmpeg_output_args
        + [output_path / f"{name_format}.{image_format}"]
    )


def dict_to_device(x, device):
    return {key: value.to(device) for key, value in x.items()}


# https://gist.github.com/wasi0013/ab73f314f8070951b92f6670f68b2d80
def download_file(url, output_path, chunk_size=4096, verbose=True):
    if verbose:
        print(f"Downloading {url}...", flush=True)
    with requests.get(url, stream=True) as source:
        with open(output_path, "wb") as output_file:
            for chunk in source.iter_content(chunk_size=chunk_size):
                if chunk:
                    output_file.write(chunk)


def get_device_description(device):
    if device == "cuda":
        return torch.cuda.get_device_name()
    else:
        return f"CPU with {torch.get_num_threads()} threads"


def get_pytorch_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def parse_patterns(pattern_file):
    patterns = []
    last_regex = None
    with open(pattern_file, "r") as text:
        for line in text:
            line = line.strip()
            if line == "":
                continue
            elif last_regex is None:
                last_regex = re.compile(line)
            else:
                patterns.append((last_regex, line))
                last_regex = None
    return patterns


def remap_weights(in_weights, patterns, verbose=False):
    n_remapped = 0
    out_weights = {}
    for in_key, weight in in_weights.items():
        out_key = in_key
        discard = False
        for regex, replacement in patterns:
            out_key, n_matches = regex.subn(replacement, out_key)
            if n_matches > 0:
                if replacement == "DISCARD":
                    discard = True
                    out_key = "DISCARD"
                n_remapped += 1
                if verbose:
                    print(f"{in_key}  ==>  {out_key}")
                break
        if not discard:
            out_weights[out_key] = weight
    return out_weights, n_remapped


def seeded_shuffle(sequence, seed):
    rng = Random()
    rng.seed(seed)
    rng.shuffle(sequence)


def set_policies(model, policy_class, **policy_kwargs):
    for gate_class in [SimpleSTGTGate, TokenDeltaGate, TokenGate]:
        for gate in model.modules_of_type(gate_class):
            gate.policy = policy_class(**policy_kwargs)


def squeeze_dict(x, dim=None):
    return {key: value.squeeze(dim=dim) for key, value in x.items()}


def tee_print(s, file, flush=True):
    print(s, flush=flush)
    print(s, file=file, flush=flush)
