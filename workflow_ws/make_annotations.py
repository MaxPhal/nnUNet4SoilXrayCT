import numpy as np
import tifffile as tiff
from skimage.filters import threshold_otsu
import napari
from os.path import join
import argparse
import json
from pathlib import Path

from __path__ import input_dir_images, input_dir_masks

input_dir_images = Path(input_dir_images)
input_dir_masks = Path(input_dir_masks)

def read_metadata(path):
    """Load the dataset metadata JSON file from the current working directory."""
    with open(path, "r") as metadata_json_file:
        metadata = json.load(metadata_json_file)
    return metadata

def load_image(input_path, sample_id):
    image_path = join(input_path, sample_id)
    grayscale_data = tiff.imread(image_path + '.tif')
    print(f"Loaded {sample_id}, shape: {grayscale_data.shape}")
    return grayscale_data

def load_annotations(output_path, sample_id):
    image_path_annotations = join(output_path, sample_id)
    saved_annotations = tiff.imread(image_path_annotations + '.tif')
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
    color_dict = {}
    for k, v in metadata["colors"].items():
        rgba = np.append(np.array(v) / 255.0, 1.0)
        color_dict[int(k)] = rgba
        #print(f"Class {k} color normalized to RGBA: {rgba}")
    return color_dict

def visualize_data(grayscale_data, annotations, color_dict):
    viewer = napari.Viewer()
    viewer.add_image(grayscale_data, name='Grayscale data')
    label_layer = viewer.add_labels(
        annotations,
        name="Annotations",
        colormap=color_dict,
        opacity=1,
        blending='additive'
    )
    label_layer.color = color_dict
    #print(label_layer.color)
    napari.run()

def save_annotations(output_path, sample_id, annotations):
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = join(output_path, sample_id + '.tif')
    tiff.imwrite(output_file, annotations.astype(np.uint8))
    print(f"Annotations saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Prepare ground truth annotations using Napari.')
    parser.add_argument('-id', type=str, required=True, help='Sample ID')
    parser.add_argument('-write', type=str, default='yes', choices=['yes', 'no'],
                        help='Whether to save annotations. "yes" overwrites previous annotations. Default: yes')
    parser.add_argument('-v', action='store_true', help='Increase output verbosity')
    args = parser.parse_args()

    # load image
    grayscale_data = load_image(input_dir_images, args.id)

    # construct path to check if annotations exist
    annotation_file = input_dir_masks / f"{args.id}.tif"

    # if annotations exist; open them; if not create them with Otsu
    if annotation_file.exists():
        annotations = load_annotations(input_dir_masks, args.id)
    else:
        annotations = apply_threshold(grayscale_data)

    # read metadata to retrieve color assignment to classes
    metadata = read_metadata(Path.cwd() / 'dataset_info.json')
    color_dict = normalize_colors(metadata)

    # visualize data with Napari
    visualize_data(grayscale_data, annotations, color_dict)

    if args.write == 'yes':
        save_annotations(input_dir_masks, args.id, annotations)
    else:
        print("Annotations were not saved")


if __name__ == "__main__":
    main()