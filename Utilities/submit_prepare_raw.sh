#!/bin/bash
#SBATCH --job-name=prepare_raw
#SBATCH --chdir=/work/phalempi
#SBATCH --output=/work/%u/%x-%A.log        #/work/user/job-name/jobid
#SBATCH --partition=compute
#SBATCH --time=60
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10                 ## out of 56
#SBATCH --mem-per-cpu=10G                  ## 10*10=200GB
#SBATCH --mail-type=BEGIN,END

##  loading Python version 3.10
module load foss/2022b Python/3.10.8

##  activating  the virtual environment
source /home/phalempi/venv310/bin/activate # modify with your paths

## declaring the environment variable
export nnUNet_raw="/work/phalempi/nnUNet_raw"

## set path to find dataset_info.json
cd /home/phalempi/nnUNet4SoilXrayCT

## Run inference with nnUNet native command
python /home/phalempi/nnUNet4SoilXrayCT/prepare_raw_data.py
