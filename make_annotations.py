import numpy as np
import tifffile as tiff
from skimage.filters import threshold_otsu
import napari
import argparse
import json
from pathlib import Path

def load_metadata(metadata_path):
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    return metadata

def load_image(input_path, sample_id):
    image_path = input_path / f'{sample_id}.tif'
    grayscale_data = tiff.imread(image_path)
    print(f"Loaded {sample_id}, shape: {grayscale_data.shape}")
    return grayscale_data

def load_annotations(output_path, sample_id):
    image_path_annotations = output_path / f'{sample_id}.tif'
    saved_annotations = tiff.imread(image_path_annotations)
    print(f"Loaded previously saved annotations of {sample_id}")
    return saved_annotations

def apply_threshold(grayscale_data):
    middle_index = grayscale_data.shape[0] // 2 - 1
    middle_slice = grayscale_data[middle_index]
    otsu_thresh = threshold_otsu(grayscale_data)
    binary_middle_slice = ((middle_slice < 220) & (middle_slice > otsu_thresh)).astype(np.uint8)
    annotations = np.zeros_like(grayscale_data, dtype=np.uint8)
    annotations[middle_index] = binary_middle_slice
    return annotations

def normalize_colors(metadata):
    color_dict = {int(k): tuple(v) for k, v in metadata["colors"].items()}
    return {k: np.array(v) / 255 for k, v in color_dict.items()}

def visualize_data(grayscale_data, annotations, color_dict):
    viewer = napari.Viewer()
    viewer.add_image(grayscale_data, name='Grayscale data')
    viewer.add_labels(annotations, name="Annotations", color=color_dict, opacity=1, blending='additive')
    napari.run()

def save_annotations(output_path, sample_id, annotations):
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / f'{sample_id}.tif'
    tiff.imwrite(output_file, annotations.astype(np.uint8))
    print(f"Annotations saved to {output_file}")

def main():

    # Parsing arguments from command line
    parser = argparse.ArgumentParser(description='This is script to prepare ground truth annotations using Napari.')
    parser.add_argument('-i', type=Path, required=True, help='Path to the input directory')
    parser.add_argument('-o', type=Path, required=True, help='Path to save the output')
    parser.add_argument('-id', type=str, required=True, help='Sample ID')
    parser.add_argument('-write', type=str, default= 'yes', required=False, help='Whether to save annotations or not - Possible answers: yes, no - /!\ yes overwrites previous annotations - Default is yes')
    parser.add_argument('-v', action='store_true', help='Increase output verbosity')
    args = parser.parse_args()

    # loading grayscale data
    grayscale_data = load_image(args.i, args.id)

    # loading annotations if they exist, otherwise applying Otsu threshold
    if (args.o / f'{args.id}.tif').exists():
        annotations = load_annotations(args.o, args.id)
    else:   
        annotations = apply_threshold(grayscale_data)
    metadata = load_metadata(Path.cwd() / 'dataset_info.json')
    color_dict = normalize_colors(metadata)
    visualize_data(grayscale_data, annotations, color_dict)

    # saving annotations if user wants to
    if args.write == 'yes':
        save_annotations(args.o, args.id, annotations)
    elif args.write == 'no':   
        print("Newly made annotations (if they were any) were not saved")
    
if __name__ == "__main__":
    main()
