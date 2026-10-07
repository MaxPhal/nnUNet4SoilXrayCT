#!/bin/bash
#SBATCH --job-name=generate_json
#SBATCH --chdir=/work/phalempi
#SBATCH --output=/work/%u/%x-%A.log    #/work/user/job-name/jobid
#SBATCH --time=5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1                
#SBATCH --mem-per-cpu=10G                  

##  loading Python version 3.10
module load foss/2022b Python/3.10.8

##  activating  the virtual environment
source /home/phalempi/venv310/bin/activate # modify with your paths

## declaring the environment variable
export nnUNet_raw="/work/phalempi/nnUNet_raw"

## set path to find dataset_info.json
cd /home/phalempi/nnUNet4SoilXrayCT/workflow_hpc

## Prepare raw data
python generate_json_file_hpc.py 
