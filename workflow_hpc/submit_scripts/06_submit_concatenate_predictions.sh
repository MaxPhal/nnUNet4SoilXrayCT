#!/bin/bash
#SBATCH --job-name=concatenate_pred
#SBATCH --chdir=/work/phalempi
#SBATCH --output=/work/%u/%x-%A-%a.log
#SBATCH --time=90
#SBATCH --mem-per-cpu=18G
#SBATCH --mail-type=BEGIN,END
#SBATCH -G nvidia-a100:1

## Loading Python 3.10.8 
module load foss/2022b Python/3.10.8

##  activating  the virtual environment
source /home/phalempi/venv310/bin/activate 

# Retrieving the name of each sample, storing it in a list and using the array job submission to submit all jobs at once
# Define the folder path
folder_path_in="/work/phalempi/predictions_split"
folder_path_out="/work/phalempi/predictions_conc"

# retrieve image name
file_list=($(ls "$folder_path_in"/*.nii.gz | xargs -n1 basename | sed -E 's/__[0-9]+__[0-9]+__[0-9]+_?\.nii\.gz$//' | sort -u))

# check list before running
echo "${#file_list[@]}"
printf '%s\n' "${file_list[@]}"

## Run inference with nnUNet native command
python concatenate_predictions_hpc.py -i "$folder_path_in" -o "$folder_path_out"  -r "${file_list[$SLURM_ARRAY_TASK_ID]}"
