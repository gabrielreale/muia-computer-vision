#!/bin/bash
##----------------------- Start job description -----------------------
#SBATCH --job-name=ResNet50_sgd
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=standard-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=8G
##------------------------ End job description ------------------------
##-------------------------- Start execution --------------------------
## Prepare environment
. visiontf_environment_setup.sh
srun python computervision/classification/inference.py ResNet50_DATAAUGM_opt_sgd_lr_0001_batch_size_64_time_202211120229
##--------------------------- End execution ---------------------------