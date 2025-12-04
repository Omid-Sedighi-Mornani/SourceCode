#!/bin/bash
#SBATCH --job-name=training
#SBATCH --partition=main
#SBATCH --time=50:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# Activate your environment and run
cd ~/SourceCode
uv run scripts/main.py --model hmm --seed 42 --state 2 --iter-sampling 5  --iter-warmup 5