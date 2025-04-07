#!/bin/bash
#SBATCH --output=run.out           # Output file
#SBATCH --error=run.err          # Error file
#SBATCH --ntasks=1                 # Number of tasks (processes)
#SBATCH --cpus-per-task=80          # Number of CPU cores per task
#SBATCH --gres=gpu:8               # Request 1 GPU
#SBATCH --time=24:00:00            # Time limit hrs:min:sec (set to what you need)
#SBATCH --partition=gpu48g

# Run Julia with the specified script and threads
JULIA_CUDA_HARD_MEMORY_LIMIT="20%" JULIA_CUDA_MEMORY_POOL="none" julia -t 4 config.jl
