#!/bin/bash
##----------------------- Start job description -----------------------
#SBATCH --job-name=example
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=standard-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
##------------------------ End job description ------------------------
##-------------------------- Start execution --------------------------
## Prepare environment
. visiontf_environment_setup.sh
srun python computervision/inference.py FFNN_DATAAUGM_opt_adam_lr_0001_lyrs_3_batch_size_64_time_202211100639
##--------------------------- End execution ---------------------------