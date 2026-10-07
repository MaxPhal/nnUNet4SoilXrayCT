import os
import json
import argparse
from os.path import join
from pathlib import Path
import nibabel as nib
import numpy as np
import tifffile

from __path__ import PATH_nnUNet_raw

def read_metadata(path):
    """Load the dataset metadata JSON file from the current working directory."""
    with open(path, "r") as metadata_json_file:
        metadata = json.load(metadata_json_file)
    return metadata

# ---------------------------------------------------------------------------
# .tif -> .nii.gz conversion, done directly in Python (no ImageJ)
# ---------------------------------------------------------------------------
def _resolution_to_spacing(res_value) -> float:
    """
    Convert a TIFF XResolution/YResolution tag value (pixels-per-unit) to a
    pixel spacing (unit-per-pixel). tifffile may return this as a (num, den)
    rational tuple or as an already-resolved float, depending on version.
    """
    if res_value is None:
        return 1.0
    if isinstance(res_value, tuple) and len(res_value) == 2:
        num, den = res_value
        if num == 0:
            return 1.0
        return den / num
    if res_value == 0:
        return 1.0
    return 1.0 / res_value


def extract_spacing_from_tif(tif: tifffile.TiffFile) -> tuple:
    """
    Extract voxel spacing (x, y, z) from a tif's embedded metadata.

    - x/y spacing come from the standard TIFF XResolution/YResolution tags.
    - z spacing comes from ImageJ's own metadata (the "spacing=..." entry in
      the ImageDescription), if present; defaults to 1.0 otherwise.

    :return: (x_spacing, y_spacing, z_spacing)
    """
    page = tif.pages[0]
    tags = page.tags

    x_res = tags["XResolution"].value if "XResolution" in tags else None
    y_res = tags["YResolution"].value if "YResolution" in tags else None

    x_spacing = _resolution_to_spacing(x_res)
    y_spacing = _resolution_to_spacing(y_res)

    ij_meta = tif.imagej_metadata or {}
    z_spacing = float(ij_meta.get("spacing", 1.0))
    unit = ij_meta.get("unit", "unknown")

    print(
        f"  Spacing from tif metadata: x={x_spacing}, y={y_spacing}, "
        f"z={z_spacing} (unit: {unit})"
    )

    return x_spacing, y_spacing, z_spacing


def read_tif_as_nib_data(tif_path: str) -> tuple:
    """
    Read a .tif image via tifffile and return (data, affine) ready for
    nib.Nifti1Image, preserving the tif's embedded x/y/z spacing.

    tifffile reads axes as (Z, Y, X); this is transposed to (X, Y, Z) to
    match this script's nibabel-based convention (the crop step below
    indexes the z-axis as shape[2]).

    :return: (img_arr, affine)
    """
    with tifffile.TiffFile(tif_path) as tif:
        img_arr = tif.asarray()
        x_spacing, y_spacing, z_spacing = extract_spacing_from_tif(tif)

    img_arr = img_arr.transpose((2, 1, 0))  # (Z, Y, X) -> (X, Y, Z)
    affine = np.diag([x_spacing, y_spacing, z_spacing, 1.0]).astype(np.float64)
    return img_arr, affine


# def img_normalize(img: np.ndarray, mean: float, std: float) -> np.ndarray:
def img_normalize(img: np.ndarray, norm_type) -> np.ndarray:
    """
    Normalize the image

    :param img:
    :param norm_type: one of [noNorm, zscore, rescale_to_0_1, rgb_to_0_1]
    :return np.ndarray:
    """
    if norm_type == "noNorm":
        return img
    elif norm_type == "zscore":
        mean_, std_ = img.mean(), img.std()
        return (img - mean_) / (max(std_, 1e-8))
    elif norm_type == "rescale_to_0_1":
        min_, max_ = img.min(), img.max()
        return (img - min_) / (max_ - min_)
    elif norm_type == "rgb_to_0_1":
        return img / 255
    else:
        raise NotImplementedError(f"Unknown normalization type: {norm_type}")

def mask_to_nnUNet(mask_data: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Convert the mask into the nnUNet Format
    For nnUNet ignore class has to be on the last index --> switch class ID 0 to the last class ID
    Reduce all other Class IDs by -1
    If Class IDs are large than the number of classes these voxel will be set to the ignore class ID

    :param mask_data:
    :param num_classes:
    :return:
    """
    mask_data[mask_data == 0] = num_classes
    x, y, z = np.where(mask_data > num_classes)
    mask_data[x, y, z] = num_classes
    mask_data -= 1
    return mask_data

if __name__ == "__main__":
    """
    Note: This differs from the original script in that it is designed to be run on a cluster and takes 
    command line arguments for a single image/annotation pair, rather than looping over a folder. 
    To launch the script, please use the bash file 01a_prepare_raw_data.sh which is located in the "Utilities" folder,
    e.g. once per SLURM array task, with the task -> image/annotation filename mapping handled there.

    This version converts the .tif pair directly to nnUNet format in Python (no ImageJ, no .hdr/.img or temp-folder
    intermediate) and saves straight into imagesTr/labelsTr.

    PARAMETERS NEEDED TO BE MANUALLY ADOPTED FOR EACH DATASET
    :param DatasetName: Name of the Dataset, can be arbitrary
    :param TaskID: Each dataset needs to have a unique ID
    :param Classes: Name of the Classes in the order of the Class IDs in the annotations. The Name
        for ID 0 (not annotated) does not have to be added.
    """

    # Parsing arguments from command line
    parser = argparse.ArgumentParser(description='This is script to prepare ground truth annotations using Napari.')
    parser.add_argument('-im', type=Path, required=True, help='Path to a single input image .tif file')
    parser.add_argument('-an', type=Path, required=True, help='Path to the matching annotation .tif file')
    parser.add_argument('-id', type=str, required=False, default=None,
                         help='Identifier for this job (e.g. image name), used for logging. '
                              'Kept for compatibility with the launcher script.')
    args = parser.parse_args()

    if args.id is not None:
        print(f"----------\nProcessing job id: {args.id}")

    # Extract metadata information from .json file
    metadata = read_metadata(Path.cwd() / 'dataset_info.json')  
    TaskID = metadata["TaskID"]
    DatasetName  = metadata["DatasetName"]
    label_names = metadata["labels"]
    Classes = list(label_names.values())
    del Classes[0] # remove the first class which is the "ToPredict" class  
    norm_type = metadata["norm_type"] # this is the normalization type

    """
    Parameters for nnUNet which are automatically adapted
    """
    num_classes = len(Classes) + 1  # +1 since we have a ignore label
    number_of_offset_layers = 48  # This parameter is needed for cropping the images - not really relevant anymore since we do not crop from the original dimension volume 
    output_folder = join(PATH_nnUNet_raw, f"Dataset{TaskID}_{DatasetName}")

    """
    Manage Folders
    """
    nnUNet_img_folder = join(output_folder, "imagesTr")
    nnUNet_mask_folder = join(output_folder, "labelsTr")
    Path(nnUNet_img_folder).mkdir(parents=True, exist_ok=True)
    Path(nnUNet_mask_folder).mkdir(parents=True, exist_ok=True)

    """
    Step1: Read mask, check label ids, convert into nnUNet format
    """
    mask_name = os.path.splitext(os.path.basename(args.an))[0]

    mask_data, mask_affine = read_tif_as_nib_data(args.an)
    mask_data = mask_data.astype(np.uint8)

    # Check if there is a Class ID outside [0:num_classes]
    min_idx, max_idx = np.min(mask_data), np.max(mask_data)
    if min_idx < 0 or max_idx >= num_classes:
        print(f"WARNING: Index ERROR in file: {args.an} - min={min_idx} max={max_idx}")
        print(f"         The corresponding Voxels will be ignored")

    # check which slices contain labeled data and crop accordingly
    _, _, z = np.where(mask_data != 0)
    z_min = max(0, np.min(z) - number_of_offset_layers)
    z_max = min(np.max(z) + number_of_offset_layers + 1, mask_data.shape[2] - 1)

    # Crop Mask and Convert to nnUNet format
    mask_data = mask_data[:, :, z_min:z_max]
    mask_data = mask_to_nnUNet(mask_data, num_classes)
    # Save Mask File
    nib.save(
        nib.Nifti1Image(mask_data, mask_affine),
        join(nnUNet_mask_folder, mask_name + ".nii.gz"),
    )
    del mask_data

    """
    Step2: Read image, crop to annotation extent, normalize, save
    """
    img_data, img_affine = read_tif_as_nib_data(args.im)

    img_data = img_data[:, :, z_min:z_max]  # crop the image to annotations
    img_data = img_normalize(img_data, norm_type)

    # Save Image File
    nib.save(
        nib.Nifti1Image(img_data, img_affine),
        join(nnUNet_img_folder, mask_name + "_0000.nii.gz"),
    )
    del img_data

    print(f"----------\nFinished: {mask_name}")