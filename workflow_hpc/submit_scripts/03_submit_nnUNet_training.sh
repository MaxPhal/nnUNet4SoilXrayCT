#!/bin/bash
#SBATCH --job-name=nnunet_training  
#SBATCH --chdir=/work/phalempi      # modify with your paths
#SBATCH --output=/work/%u/%x-%A-%a.log
#SBATCH --time=4300                 # modify with your time limit (in min); < 3 days (on EVE)
#SBATCH --nodes=1                   # one GPU per node
#SBATCH --ntasks=1                  # one training fold per task
#SBATCH --cpus-per-task=36          # 36 CPU from 56 CPU available
#SBATCH --mem-per-cpu=5G            # 36*5=180GB so (180/512)+-= 35.2% of total RAM of the node
#SBATCH --mail-type=BEGIN,END       # modify with your mail preferences
#SBATCH -G nvidia-a100             # modify with your GPU preferences

##  loading Python version 3.10
module load foss/2022b Python/3.10.8

##  activating  the virtual environment
source /home/phalempi/venv310/bin/activate # modify with your paths

## declaring the environment variable
export nnUNet_preprocessed="/work/phalempi/nnUNet_preprocessed" # modify with your path
export nnUNet_results="/work/phalempi/nnUNet_results" # modify with your path

## Setting the correct number of workers used for data augmentation (for training only). This is now directly passed from the SBATCH parameters
export nnUNet_n_proc_DA=${SLURM_CPUS_PER_TASK:-1}

## Run training with nnUNet native command
nnUNetv2_train 777 3d_fullres $SLURM_ARRAY_TASK_ID -tr nnUNetTrainer_betterIgnoreSampling -p nnUNetResEncUNetPlans_M40G
