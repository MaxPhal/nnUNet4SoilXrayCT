import json
import numpy as np
import pandas as pd

# Notes on file path
# All summary and data set paths should be in the same folder "input"
path_to_folder = 'C:/Users/phalempi/Desktop/Image_Analysis/nnUnet/dice_score'
# Define the desired dataset to be analyzed
Dataset = '304' 
# List of file paths - these are the files named "summary.json" in the validation folder of each fold
file_paths = [
    path_to_folder+'/input/Dataset'+Dataset+'_fold0_summary_val.json',
    path_to_folder+'/input/Dataset'+Dataset+'_fold1_summary_val.json',
    path_to_folder+'/input/Dataset'+Dataset+'_fold2_summary_val.json',
    path_to_folder+'/input/Dataset'+Dataset+'_fold3_summary_val.json',
    path_to_folder+'/input/Dataset'+Dataset+'_fold4_summary_val.json',
]

# Initialize lists to store Dice values
foreground_mean_dice_values = []
mean_dice_values_list = []

# Loop through each file path
for file_path in file_paths:
    # Open and read the JSON file
    with open(file_path, 'r') as file:
        json_data = json.load(file)  # Load the JSON content
        
        # Extract the Dice values
        foreground_mean_dice = json_data["foreground_mean"]["Dice"]
        mean_dice_values = {key: value["Dice"] for key, value in json_data["mean"].items()}
        
        # Store the results
        foreground_mean_dice_values.append(foreground_mean_dice)
        mean_dice_values_list.append(mean_dice_values)

# Calculate mean and standard error for foreground_mean_dice
foreground_mean_dice_mean = round(np.mean(foreground_mean_dice_values), 2)
foreground_mean_dice_se = round(np.std(foreground_mean_dice_values, ddof=1) / np.sqrt(len(foreground_mean_dice_values)),4)

# Prepare to calculate mean and standard error for mean_dice_values
mean_dice_values_keys = mean_dice_values_list[0].keys()
mean_dice_results = {}

# Calculate mean and standard error for each key in mean_dice_values
for key in mean_dice_values_keys:
    # Collect values across all files for the current key
    values = [mean_dice_values[key] for mean_dice_values in mean_dice_values_list]
    mean_dice_mean = np.mean(values)
    mean_dice_se = np.std(values, ddof=1) / np.sqrt(len(values))
    
    # Store results
    mean_dice_results[key] = {
        'mean': round(mean_dice_mean, 2),
        'standard_error': round(mean_dice_se, 4)
    }
    
# Read the label file. This is the file named "summary" in the validation folder of each fold

label_file= path_to_folder+'/input/Dataset'+Dataset+'_labels.json'  # Load the JSON content
# Open and read the JSON file
with open(label_file, 'r') as file:
    json_data = json.load(file)  # Load the JSON content

# Extract the labels into a list
labels_list = list(json_data["labels"].keys())
labels_list[0] = 'pores' ## for Dataset304, first class was "pores" 
# Here I am (almost 100%) sure that "foreground_mean" is actually the matrix ## here it should be replaced by the name of the first class defined in line 162 of preprocessing_nnUNet_train.py

# Initialize lists for classes, mean, and SE
classes = labels_list[:-1]  # [:-1] to remove last entry of list (which was the ignore label)
mean_values = []
mean_values.append(foreground_mean_dice_mean)  # Adds to the end of the list
se_values = []
se_values.append(foreground_mean_dice_se)  # Adds to the end of the list

# Fill in the mean and SE for each class (excluding 'matrix' for now)
for key, results in mean_dice_results.items():
    mean_values.append(results['mean'])
    se_values.append(results['standard_error'])

# Create the DataFrame
df_summary = pd.DataFrame({
    'classes': classes,
    'mean': mean_values,
    'se': se_values
})

# Output the DataFrame
print(df_summary)
df_summary.to_csv(path_to_folder + '/output/dice_results'+Dataset+'.csv', index=False)

