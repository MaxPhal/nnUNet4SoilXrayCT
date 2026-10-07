import glob
import os
from os.path import join
from pathlib import Path
from typing import List
import nibabel as nib
import numpy as np
import tifffile
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from tqdm import tqdm
from make_annotations import read_metadata

from __path__ import PATH_nnUNet_raw, input_dir_images, input_dir_masks

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
    match this script's nibabel-based convention (Step 3 below indexes the
    z-axis as shape[2]).

    :return: (img_arr, affine)
    """
    with tifffile.TiffFile(tif_path) as tif:
        img_arr = tif.asarray()
        x_spacing, y_spacing, z_spacing = extract_spacing_from_tif(tif)

    img_arr = img_arr.transpose((2, 1, 0))  # (Z, Y, X) -> (X, Y, Z)
    affine = np.diag([x_spacing, y_spacing, z_spacing, 1.0]).astype(np.float64)
    return img_arr, affine


def get_img_file(mask_name: str, img_files: List[str], img_postfix: str) -> str:
    """
    Get the image file which corresponds to the mask_name

    :param mask_name: name of the current mask file (without extension)
    :param img_files: list of all candidate image files (any extension)
    :param img_postfix: postfix of the image files to match mask and image files
    :return str:
    """
    img_names = [
        os.path.splitext(os.path.basename(img_file))[0].replace(img_postfix, "")
        for img_file in img_files
    ]
    for i, name in enumerate(img_names):
        if name in mask_name:
            return img_files[i]
    return None

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
    Note: This script is intented to be run on a workstation and not on a cluster. It is designed to prepare the raw data for training with nnUNet.
    It takes the input images and annotations, converts them directly from .tif to the nnUNet format (cropped, normalized), and saves them 
    straight into imagesTr/labelsTr - no intermediate temp folder is used. 
    The script also handles the creation of the necessary directory structure for nnUNet training and generates a dataset.json file which is required for nnUNet 
    to recognize the dataset.

    PARAMETERS NEEDED TO BE MANUALLY ADOPTED FOR EACH DATASET
    :param input_dir_images: Path to the folder which contains the images in the .tif file format -- this is now read from the __path__ file
    :param input_dir_masks: Path to the folder which contains the annotations in the .tif file format -- this is now read from the __path__ file
        Requirements:
        - label 0 are the voxels which are not annotated and should be ignored
        - label 1 should be the soil matrix
    :param DatasetName: Name of the Dataset, can be arbitrary
    :param TaskID: Each dataset needs to have a unique ID
    :param Classes: Name of the Classes in the order of the Class IDs in the annotations. The Name
        for ID 0 (not annotated) does not have to be added.
    :param img_file_postfix: to match the images with the annotations. E.g. image name 07_norm.mha
        and the annotation has the name 07_annotations_v1.mha. To match both the image postfix has
        to be removed which is "_norm" is this case. If image and annotations are the same this can
        also be empty ("").
    """

    # Extract metadata information from .json file
    metadata = read_metadata(Path.cwd() / 'dataset_info.json')
    TaskID = metadata["TaskID"]
    DatasetName  = metadata["DatasetName"]
    label_names = metadata["labels"]
    Classes = list(label_names.values())
    del Classes[0] # remove the first class which is the "ToPredict" class  
    norm_type = metadata["norm_type"] # this is the normalization type
    img_file_postfix = "" # empty if image and annotations have the same name otherwise something like: "_norm" // this works if img file has a suffixe, not if the annotations have a suffix

    """
    Parameters for nnUNet which are automatically adapted
    """
    num_classes = len(Classes) + 1  # +1 since we have a ignore label
    number_of_offset_layers = 48  # This parameter is needed for cropping the images
    output_folder = join(PATH_nnUNet_raw, f"Dataset{TaskID}_{DatasetName}")

    """
    Manage Folders
    """
    nnUNet_img_folder = join(output_folder, "imagesTr")
    nnUNet_mask_folder = join(output_folder, "labelsTr")
    Path(nnUNet_img_folder).mkdir(parents=True, exist_ok=True)
    Path(nnUNet_mask_folder).mkdir(parents=True, exist_ok=True)

    """
    Step1: Convert each annotation/image .tif pair directly into nnUNet
    format, straight into imagesTr/labelsTr - no intermediate temp folder
    """
    mask_tif_files = sorted(glob.glob(join(input_dir_masks, "*.tif")))
    img_tif_files = sorted(glob.glob(join(input_dir_images, "*.tif")))
    print(f"----------\n{len(mask_tif_files)} annotation files found")
    print(f"----------\n{len(img_tif_files)} grayscale images found")

    for mask_tif_file in tqdm(mask_tif_files, desc="Convert File to nnUNet Format"):
        """
        Find corresponding image file for the mask file
        """
        mask_name = os.path.splitext(os.path.basename(mask_tif_file))[0] # returns image_ID without extension

        img_tif_file = get_img_file(mask_name, img_tif_files, img_file_postfix)
        print(img_tif_file)

        if img_tif_file is None:
            print(f"ERROR: No Image file was found for {mask_tif_file}\n       Skipping {mask_tif_file}")
            continue

        """
        Read mask, check label ids, convert into nnUNet format
        """
        mask_data, mask_affine = read_tif_as_nib_data(mask_tif_file)
        mask_data = mask_data.astype(np.uint8)

        # Check if there is a Class ID outside [0:num_classes]
        min_idx, max_idx = np.min(mask_data), np.max(mask_data)
        if min_idx < 0 or max_idx >= num_classes:
            print(f"WARNING: Index ERROR in file: {mask_tif_file} - min={min_idx} max={max_idx}")
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
        Read image, crop to annotation extent, normalize, save
        """
        img_data, img_affine = read_tif_as_nib_data(img_tif_file)

        img_data = img_data[:, :, z_min:z_max]  # crop the image to annotations
        img_data = img_normalize(img_data, norm_type)

        # Save Image File
        nib.save(
            nib.Nifti1Image(img_data, img_affine),
            join(nnUNet_img_folder, mask_name + "_0000.nii.gz"),
        )
        del img_data

    """
    Step2: Create the dataset.json which is needed for nnUNet and contains information about the dataset
    Note: Here the dataset.json is created after all images and annotations have been processed and saved in
    the nnUNet format. This is because the dataset.json requires information about the number of training images, 
    which can only be determined after processing all files.
    """
    Classes[0] = "background"  # first class has to be named background, corresponds to soil matrix
    Classes.append("ignore")  # ignore label has to be on the last position
    labels = {name: i for i, name in enumerate(Classes)}
    generate_dataset_json(
        output_folder=output_folder,
        channel_names={0: norm_type},
        labels=labels,
        num_training_cases=len(glob.glob(join(nnUNet_img_folder, "*.nii.gz"))),
        file_ending=".nii.gz",
        dataset_name=DatasetName,
    )