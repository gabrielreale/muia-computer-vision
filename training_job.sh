#!/bin/bash
##----------------------- Start job description -----------------------
#SBATCH --job-name=example
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=standard-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=16G
##------------------------ End job description ------------------------
##-------------------------- Start execution --------------------------
srun python computervision/training.py 
##--------------------------- End execution ---------------------------