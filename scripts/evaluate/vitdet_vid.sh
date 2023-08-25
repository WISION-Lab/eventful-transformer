#!/usr/bin/env bash

# Slurm wrapper for scripts/evaluate/vitdet_vid.py. Usage:
# sbatch -J <job-name> ./scripts/evaluate/vitdet_vid.sh
# where <job-name> is the name of the config in configs/evaluate/vitdet_vid.

# To override the time limit, use the -t/--time command-line argument.

# To request a specific GPU, use the argument --gres=gpu:<type>:1 with <type>
# replaced by the type of GPU (e.g, a100).

#SBATCH --cpus-per-task=16
#SBATCH --output=slurm/%x.txt
#SBATCH --gres=gpu:1
#SBATCH --mem=48GB
#SBATCH --partition=research
#SBATCH --time=4-00:00:00

./scripts/evaluate/vitdet_vid.py "$SLURM_JOB_NAME"
