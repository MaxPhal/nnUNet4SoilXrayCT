import argparse
import glob
import os
from os.path import join

import nibabel as nib
import numpy as np
from tqdm import tqdm

from preprocessing_nnUNet_train import img_normalize, convert_tif_to_hdr

def convert_hdr_to_nii_normalize(input_dir: str,norm_type:str) -> None:
    """
    Convert the .hdr/.img files to .nii.gz and save them in the same directory.
    The Image is normalized and renamed to be in the correct format for nnUNet
    Afterward delete the .hdr/.img files.

    :param input_dir: directory which contains the .hdr/.img files
    :param norm_type: one of [noNorm, zscore, rescale_to_0_1, rgb_to_0_1]

    :return:
    """
    hdr_files = glob.glob(join(input_dir, "*.hdr"))

    for hdr_file in tqdm(hdr_files, desc="Process Image to .nii.gz"):
        # Load the file
        img = nib.load(hdr_file)
        img_arr = img.get_fdata().astype(np.uint8)[:, :, :, 0]

        # Normalize the image
        img_arr = img_normalize(img_arr, norm_type)

        # Save the file as .nii.gz
        nib.save(
            nib.Nifti1Image(img_arr, img.affine, img.header),
            hdr_file.replace(".hdr", "_0000.nii.gz"),
        )

        # Clear RAM and delete the .hdr/.img files
        del img, img_arr
        os.remove(hdr_file)
        os.remove(hdr_file.replace(".hdr", ".img"))


if __name__ == "__main__":
    """
    This script convert each image file in the input folder from .tif to .nii.gz, normalizes the
    image and saves it to the the output folder
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        help="Path to the folder which contains the images in the .mha file format",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Path to the folder into which the processed images should be saved in",
    )
    parser.add_argument(
        "-n",
        "--norm",
        default="zscore",
        help="normalization type, one of [noNorm, zscore, rescale_to_0_1, rgb_to_0_1]",
    )
    args = parser.parse_args()

    input_dir_images = args.input
    output_dir_images = args.output
    #convert_mha_to_hdr(input_dir_images, output_dir_images)
    convert_tif_to_hdr(input_dir_images, output_dir_images)
    convert_hdr_to_nii_normalize(output_dir_images, args.norm)
