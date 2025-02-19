import numpy as np
import tifffile as tiff
from skimage.filters import threshold_otsu
import napari
import argparse
import json
import os 

# Create the ArgumentParser object
parser = argparse.ArgumentParser(description='Argument parsing')

# Define command-line arguments
parser.add_argument('-i', type=str, required=True, help='Path to the input file')
parser.add_argument('-o', type=str, required=True, help='Path to save the output')
parser.add_argument('-id', type=str, required=True, help='Sample ID')
parser.add_argument('-v', action='store_true', help='Increase output verbosity')

# Parse arguments from the terminal
args = parser.parse_args()

# Load the 3D .tif file
grayscale_data = tiff.imread(args.i + '/' + args.id + '.tif')

# Check the shape of the 3D image (should be something like (z, y, x))
print(f"Image {args.id} is loaded")
print(f"Image shape: {grayscale_data.shape}")

# Get the middle slice index
middle_index = (grayscale_data.shape[0] // 2)-1
middle_slice = grayscale_data[middle_index]

# Calculate Otsu threshold on the whole subvolume
otsu_thresh = threshold_otsu(grayscale_data)
# Apply Otsu thresholding
binary_middle_slice = ((middle_slice < 220) & (middle_slice > otsu_thresh)).astype(np.uint8)

# Create an empty 3D stack of zeros
annotations = np.zeros_like(grayscale_data, dtype=np.uint8)

# Insert the binary middle slice into the correct position
annotations[middle_index] = binary_middle_slice

# Load the JSON file
cwd = os.getcwd()
with open(cwd + '/metadata.json', "r") as json_file:
    data = json.load(json_file)

# Extract label names
label_names = data["labels"]

# Extract color dictionary (convert keys back to integers)
color_dict = {int(k): tuple(v) for k, v in data["colors"].items()}

# Normalize the RGB values for Napari (values must be between 0 and 1)
normalized_color_dict = {k: (np.array(v) / 255) for k, v in color_dict.items()}

# Create properties dictionary for label names
properties = {'index': list(label_names.keys()), 'name': list(label_names.values())}

# Launch Napapi 
viewer = napari.Viewer()
viewer.add_image(grayscale_data, name='Grayscale data')

# Add label layer with properties
label_layer = viewer.add_labels(annotations, 
                                name="Annotations", 
                                properties=properties, 
                                color=normalized_color_dict, 
                                opacity=1,
                                blending='additive')

# Assign label names explicitly
label_layer.metadata['label_names'] = label_names  # Store for later use

napari.run()

# Save annotations as 3D .tif 8-bit format (values between 0-255)
annotations_8bit = annotations.astype(np.uint8)
    
# Save the 3D .tif file
tiff.imwrite(args.o + '/' + args.id + '_annotations.tif', annotations_8bit)
print(f"Annotations of {args.id} saved to {args.o}")