#!/bin/bash
#SBATCH --job-name=prepare_raw
#SBATCH --chdir=/work/phalempi
#SBATCH --output=/work/%u/%x-%A-%a.log        #/work/user/job-name/jobid
#SBATCH --partition=compute
#SBATCH --time=20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1                  ## 1 CPU per image
#SBATCH --mem-per-cpu=50G                  ## 10*10=200GB

##  loading Python version 3.10
module load foss/2022b Python/3.10.8

##  activating  the virtual environment
source /home/phalempi/venv310/bin/activate # modify with your paths

## declaring the environment variable
export nnUNet_raw="/work/phalempi/nnUNet_raw"

## set path to find dataset_info.json and __path__.py
cd /home/phalempi/nnUNet4SoilXrayCT

## retrieve file paths from __path__.py (in pwd)
input_dir_images=$(python3 -c "from __path__ import input_dir_images; print(input_dir_images)")
input_dir_masks=$(python3 -c "from __path__ import input_dir_masks; print(input_dir_masks)")

# Initialize an empty list
file_list=()

# Loop through all files in the folder and add them to the list
# this works assuming that grayscale and annotations have the same name but are located in different folders!
for file in "$input_dir_images"/*; do
    # Check if it's a file (not a directory)
    if [ -f "$file" ]; then
        file_list+=("$(basename "$file")")
    fi
done

## Prepare raw data
python /home/phalempi/nnUNet4SoilXrayCT/02_prepare_raw_data_for_training_hpc.py -im "$input_dir_images"/"${file_list[$SLURM_ARRAY_TASK_ID]}" -an "$input_dir_masks"/"${file_list[$SLURM_ARRAY_TASK_ID]}" -id "${file_list[$SLURM_ARRAY_TASK_ID]}"
