#!/bin/bash

# Initialize variables
folder_path_in=""
folder_path_out=""

# Parse input arguments
while getopts "i:o:" opt; do
  case $opt in
    i) folder_path_in="$OPTARG" ;;  # Input folder
    o) folder_path_out="$OPTARG" ;; # Output folder
    \?) echo "Usage: $0 -i <input_folder> -o <output_folder>"
        exit 1 ;;
  esac
done

# Validate input arguments
if [[ -z "$folder_path_in" || -z "$folder_path_out" ]]; then
    echo "Error: Both -i (input folder) and -o (output folder) are required."
    echo "Usage: $0 -i <input_folder> -o <output_folder>"
    exit 1
fi

# Ensure input folder exists
if [[ ! -d "$folder_path_in" ]]; then
    echo "Error: Input folder does not exist: $folder_path_in"
    exit 1
fi

# Create output folder if it does not exist
mkdir -p "$folder_path_out"

# Initialize an empty list
file_list=()

# Loop through all files in the input folder and add them to the list
for file in "$folder_path_in"/*; do
    if [[ -f "$file" ]]; then
        file_list+=("$(basename "$file")")
    fi
done

# Loop through the list of file names and create a folder for each
for file_name in "${file_list[@]}"; do
    new_folder="$folder_path_out/${file_name}"
    mkdir -p "$new_folder"
    echo "Created folder: $new_folder"

    # Move the file into the new folder
    mv "$folder_path_in/$file_name" "$new_folder/"
    echo "Moved file: $file_name to $new_folder"
done