import glob
import os
import shutil
import subprocess
from os.path import join, split
from pathlib import Path
from typing import List

import nibabel as nib
import numpy as np
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from tqdm import tqdm

from __path__ import PATH_ImageJ, PATH_nnUNet_raw


def convert_mha_to_hdr(input_dir: str, output_dir: str) -> None:
    """
    Use ImageJ and the convert_mha_to_img macro script to convert .mha files to .hdr & .img files

    :param input_dir: path to the input folder which contains the .mha files
    :param output_dir: path to the folder in which the output should be saved in
    :return:
    """
    # os.makedirs(output_dir, exist_ok=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print("Process .mha to .hdr")
    subprocess.Popen(
        rf'{PATH_ImageJ} --headless -macro convert_mha_to_img "{input_dir}--{output_dir}"',
        # fr'{PATH_ImageJ} -macro convert_mha_to_img "{input_dir}--{output_dir}"', # it helps to put it in macro mode to debug
        shell=True,
    ).wait()
    
def convert_tif_to_hdr(input_dir: str, output_dir: str) -> None:
    """
    Use ImageJ and the convert_tif_to_img macro script to convert .tif files to .hdr & .img files

    :param input_dir: path to the input folder which contains the .mha files
    :param output_dir: path to the folder in which the output should be saved in
    :return:
    """
    # os.makedirs(output_dir, exist_ok=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print("Process .tif to .hdr")
    subprocess.Popen(
        rf'{PATH_ImageJ} --headless -macro convert_tif_to_img "{input_dir}--{output_dir}"',
        #fr'{PATH_ImageJ} -macro convert_mha_to_img "{input_dir}--{output_dir}"', # it helps to put it in macro mode to debug
        shell=True,
    ).wait()


def convert_hdr_to_nii(input_dir: str, is_mask: bool = False, num_classes: int = None) -> None:
    """
    Convert the .hdr/.img files to .nii.gz and save them in the same directory.
    Afterward delete the .hdr/.img files.
    If is_mask check if all Class IDs in the annotation are between 0 and num_classes

    :param input_dir: directory which contains the .hdr/.img files
    :param is_mask: if the file is a mask, and it should be checked if label ids are valid
    :param num_classes: if is_mask is set, the number of classes is needed.
    :return:
    """
    hdr_files = glob.glob(join(input_dir, "*.hdr"))

    for hdr_file in tqdm(hdr_files, desc="Process File to .nii.gz"):
        # Load the file
        img = nib.load(hdr_file)
        img_arr = img.get_fdata().astype(np.uint8)[:, :, :, 0]
        # Check if there is a Class ID outside [0:num_classes]
        if is_mask:
            min_idx, max_idx = np.min(img_arr), np.max(img_arr)
            if min_idx < 0 or max_idx >= num_classes:
                print(f"WARNING: Index ERROR in file: {hdr_file} - min={min_idx} max={max_idx}")
                print(f"         The corresponding Voxels will be ignored")

        # Save the file as .nii.gz
        nib.save(
            nib.Nifti1Image(img_arr, img.affine, img.header),
            hdr_file.replace(".hdr", ".nii.gz"),
        )

        # Clear RAM and delete the .hdr/.img files
        del img, img_arr
        os.remove(hdr_file)
        os.remove(hdr_file.replace(".hdr", ".img"))


def get_img_file(mask_name: str, img_files: List[str], img_postfix: str) -> str:
    """
    Get the image file which corresponds to the mask_name

    :param mask_name: name of the current mask file
    :param img_files: list of all image files
    :param img_postfix: postfix of the image files to match mask and image files
    :return str:
    """
    img_names = [split(img_file)[-1].replace(img_postfix + ".nii.gz", "") for img_file in img_files]
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
    PARAMETERS NEEDED TO BE MANUALLY ADOPTED FOR EACH DATASET
    :param input_dir_images: Path to the folder which contains the images in the .mha file format
    :param input_dir_masks: Path to the folder which contains the annotations in the .mha file format
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
    # Ubuntu
    # input_dir_images = ("/home/l727r/Desktop/UFZ_2022_CTPoreRootSegmentation/ufz_2023_conversion_test/raw/images")
    # input_dir_masks = "/home/l727r/Desktop/UFZ_2022_CTPoreRootSegmentation/ufz_2023_conversion_test/raw/annotations"
    
    # Windows
    input_dir_images = r"F:\phalempin\grayscale_data\dataset3_icecores\training_gray"
    input_dir_masks = r"F:\phalempin\annotations\dataset3_icecores\third_round"

    DatasetName = "Ice_cores_3rdround"
    TaskID = 304
    Classes = [
        "pores",
        "mineral",
        "ice",
        "POM",
    ]
    norm_type = "noNorm" # one of [noNorm, zscore, rescale_to_0_1, rgb_to_0_1] with default==zscore
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
    temp_img_folder = join(output_folder, "temp_images_nii")
    temp_mask_folder = join(output_folder, "temp_labels_nii")

    nnUNet_img_folder = join(output_folder, "imagesTr")
    nnUNet_mask_folder = join(output_folder, "labelsTr")
    # os.makedirs(nnUNet_img_folder, exist_ok=True)
    # os.makedirs(nnUNet_mask_folder, exist_ok=True)
    Path(nnUNet_img_folder).mkdir(parents=True, exist_ok=True)
    Path(nnUNet_mask_folder).mkdir(parents=True, exist_ok=True)

    """
    Step1: Convert annotation files from .tif to .nii.gz 
    """
    #convert_mha_to_hdr(input_dir_mask, temp_img_folder) # uncomment if annotation files are in .mha format
    convert_tif_to_hdr(input_dir_masks, temp_mask_folder) # with the new napari workflow for annotations, the images are saved in .tif automatically
    convert_hdr_to_nii(temp_mask_folder, True, num_classes)
    """
    Step2: Convert image files from .tif to .nii.gz 
    """
    #convert_mha_to_hdr(input_dir_images, temp_img_folder) # uncomment if input grayscale data are in .mha format
    convert_tif_to_hdr(input_dir_images, temp_img_folder) 
    convert_hdr_to_nii(temp_img_folder)
    """
    Step3: Convert everything into nnUNet format
    """
    mask_files = glob.glob(join(temp_mask_folder, "*.nii.gz"))
    img_files = glob.glob(join(temp_img_folder, "*.nii.gz"))

    for mask_file in tqdm(mask_files, desc="Convert File to nnUNet Format"):
        """
        Find corresponding image file for the mask file
        """
        mask_name = split(mask_file)[-1].replace(".nii.gz", "")
        img_file = get_img_file(mask_name, img_files, img_file_postfix)
        if img_file is None:
            print(f"ERROR: No Image file was found for {mask_file}\n       Skipping {mask_file}")
            continue

        """
        Convert mask into nnUNet format
        """
        mask = nib.load(mask_file)
        mask_data = mask.get_fdata().astype(np.uint8)

        # check which slices contain labeled data and crop accordingly
        _, _, z = np.where(mask_data != 0)
        z_min = max(0, np.min(z) - number_of_offset_layers)
        z_max = min(np.max(z) + number_of_offset_layers + 1, mask_data.shape[2] - 1)

        # Crop Mask and Convert to nnUNet format
        mask_data = mask_data[:, :, z_min:z_max]
        mask_data = mask_to_nnUNet(mask_data, num_classes)
        # Save Mask File
        nib.save(
            nib.Nifti1Image(mask_data, mask.affine, mask.header),
            join(nnUNet_mask_folder, mask_name + ".nii.gz"),
        )
        del mask, mask_data

        """
        Convert image into nnUNet format
        """
        img = nib.load(img_file)
        img_data = img.get_fdata()

        # mean, std = img_data.mean(), img_data.std()
        img_data = img_data[:, :, z_min:z_max] # this was commented in the downloaded script! It is needs to be uncommented so that cropping occurs for the grayscale also so that dimensions match
        img_data = img_normalize(img_data, norm_type)

        nib.save(
            nib.Nifti1Image(img_data, img.affine, img.header),
            join(nnUNet_img_folder, mask_name + "_0000.nii.gz"),
        )
        del img, img_data

    """
    Step4: Create the dataset.json which is needed for nnUNet and contains information about the dataset
    """
    Classes[0] = "background"  # first class has to be named background, corresponds to soil matrix
    Classes.append("ignore")  # ignore label has to be on the last position
    labels = {name: i for i, name in enumerate(Classes)}
    generate_dataset_json(
        output_folder=output_folder,
        channel_names={0: "noNorm"},
        labels=labels,
        num_training_cases=len(glob.glob(join(nnUNet_img_folder, "*.nii.gz"))),
        file_ending=".nii.gz",
        dataset_name=DatasetName,
    )

    """
    Step5: Clean up and delete the temporal folders
    """
    shutil.rmtree(temp_img_folder)
    shutil.rmtree(temp_mask_folder)
