#!/bin/bash
#SBATCH --job-name=nnunet_preprocessing
#SBATCH --chdir=/work/phalempi
#SBATCH --output=/work/%u/%x-%A.log
#SBATCH --partition=compute
#SBATCH --time=20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10                 ## out of 56
#SBATCH --mem-per-cpu=20G                  ## 10*20=100GB out of 470
#SBATCH --mail-type=BEGIN,END

##  loading Python version 3.10
module load foss/2022b Python/3.10.8

##  activating  the virtual environment
source /home/phalempi/venv310/bin/activate # modify with your paths

## declaring the environment variable
export nnUNet_raw="/work/phalempi/nnUNet_raw"
export nnUNet_preprocessed="/work/phalempi/nnUNet_preprocessed"

## Run inference with nnUNet native command
nnUNetv2_plan_and_preprocess -d 777 -c 3d_fullres -np 8 -npfp 8

## Planning experiment with new ResEnc Presets using target GPU memory! 
nnUNetv2_plan_experiment -d 777 -pl nnUNetPlannerResEncM -gpu_memory_target 40 -overwrite_plans_name nnUNetResEncUNetPlans_M40G

