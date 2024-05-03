#!/bin/bash

#SBATCH --job-name=9peat-1kpop     # Job name, you can change it to whatever you want
#SBATCH --output=run.out           # Output file
#SBATCH --error=run.err          # Error file
#SBATCH --ntasks=1                 # Number of tasks (processes)
#SBATCH --cpus-per-task=20          # Number of CPU cores per task
#SBATCH --gres=gpu:4               # Request 1 GPU
#SBATCH --time=24:00:00            # Time limit hrs:min:sec (set to what you need)

# Run Julia with the specified script and threads
julia -t 4 config.jl
