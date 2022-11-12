#!/bin/bash
##----------------------- Start job description -----------------------
#SBATCH --job-name=inference
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=standard-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=4G
##------------------------ End job description ------------------------
##-------------------------- Start execution --------------------------
## Prepare environment
. visiontf_environment_setup.sh
srun python computervision/inference.py VGG19_DATAAUGM_opt_adam_lr_0001_batch_size_64_time_202211120012
##--------------------------- End execution ---------------------------