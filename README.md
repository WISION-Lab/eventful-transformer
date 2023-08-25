## Disclaimer

This is research-grade code, so it's possible you will encounter some hiccups. [Contact me](https://github.com/mattdutson/) if you encounter problems or if the documentation is unclear, and I will do my best to help.

## TLDR

Most of the interesting code (implementation of our core contributions) is in `eventful_transformer/blocks.py`, `eventful_transformer/modules.py`, and `eventful_transformer/policies.py`.

## Dependencies

Dependencies are managed using Conda. The environment is defined in `environment.yml`.

To create the environment, run:
```
conda env create -f environment.yml
```
Then activate the environment with:
```
conda activate eventful-transformer
```

## Running Scripts

Scripts should be run from the repo's base directory.

Many scripts expect a `.yml` configuration file as a command-line argument. These configuration files are in `configs`. The structure of the `configs` folder is set to mirror the structure of the `scripts` folder. For example, to run the `base_672` evaluation for the ViTDet VID model:
```
./scripts/evaluate/vitdet_vid.py ./configs/evaluate/vitdet_vid/base_672.yml
```

## Weights

Weights for the ViViT action recognition model are available [here](https://github.com/alibaba-mmai-research/TAdaConv/blob/main/MODEL_ZOO.md). Weights for the ViTDet object detection model are available [here](https://github.com/facebookresearch/detectron2/tree/main/projects/ViTDet).

The weight names need to be remapped to work with this codebase. To remap the ViViT weights, run:
```
./scripts/convert/vivit.py <old_weights> <new_weights> ./configs/convert/vivit_b.txt
```
with `<old_weights>` and `<new_weigtht>` replaced by the path of the downloaded weights and the path where the converted weights should be saved, respectively.

To remap the ViTDet weights, run:
```
./scripts/convert/vitdet.py <old_weights> <new_weights> ./configs/convert/vitdet_b.txt
```

## Data

The `datasets` folder defines PyTorch `Dataset` classes for Kinetics-400, VID, and EPIC-Kitchens.

The Kinetics-400 class will automatically download and prepare the dataset on first use.

VID requires a manual download. Download `vid_data.tar` from [here](https://drive.google.com/drive/folders/1tNtIOYlCIlzb2d_fCsIbmjgIETd-xzW-) and place it at `./data/vid/data.tar`. The VID class will take care of unpacking and preparing the data on first use.

EPIC-Kitchens also requires a manual download. Download the videos from [here](https://drive.google.com/drive/folders/1OKJpgSKR1QnWa2tMMafknLF-CpEaxDbY) and place them in `./data/epic_kitchens/videos`. Download the labels `EPIC_100_train.csv` and `EPIC_100_validation.csv` from [here](https://github.com/epic-kitchens/epic-kitchens-100-annotations) and place them in `./data/epic_kitchens`. The EPICKitchens class will prepare the data on first use.

## Other Setup

Scripts assume that the current working directory is on the Python path. In the Bash shell, run
```
export PYTHONPATH="$PYTHONPATH:."
```
Or in the Fish shell:
```
set -ax PYTHONPATH .
```

## Code Style

Format all code using [Black](https://black.readthedocs.io/en/stable/). Use a line limit of 88 characters (the default). To format a file, use the command:
```
black <FILE>
```
