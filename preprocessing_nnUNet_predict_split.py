import argparse
import json
import os
from os.path import join
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def split_files(
    img_folder: str,
    output_folder: str,
    model_path: str,
    num_splits: int = 8,
    axis: int = 2,
):
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    if model_path is not None:
        with open(join(model_path, "plans.json"), "r") as file:
            plans = json.load(file)
        patch_size = plans["configurations"]["3d_fullres"]["patch_size"][::-1]
        target_spacing = plans["configurations"]["3d_fullres"]["spacing"][::-1]
    else:
        print(
            "Warning: No model_path was given, default patch_size and target_spacing are taken, for exact values give the model path"
        )
        patch_size = [224, 224, 48]
        target_spacing = [1.0, 1.0, 1.0]

    print(f"Model Parameters:")
    print(f"Patch Size:     {patch_size}")
    print(f"Target Spacing: {target_spacing}")

    images = os.listdir(img_folder)
    images = [image for image in images if image.endswith("_0000.nii.gz")]
    print(f"----------\n{len(images)} Images found")
    for image in images:
        print("----------")
        print(f"Image File:    {image}")
        img = sitk.ReadImage(join(img_folder, image))
        img_shape = img.GetSize()
        img_spacing = img.GetSpacing()
        img_direction = img.GetDirection()
        img_origin = img.GetOrigin()

        print(f"Image Shape:   {img_shape}")
        print(f"Image Spacing: {img_spacing}")

        original_patch_size = (
            patch_size * np.array(target_spacing) / np.array(img_spacing)
        )
        overlap = np.ceil(original_patch_size / 2)
        crop_size = np.array(img_shape)
        crop_size[axis] = np.ceil(img_shape[axis] / num_splits)

        print(f"Original Patch Size:   {original_patch_size}")
        print(f"Patch Overlap:   {overlap}")
        print(f"Base Crop Size:   {crop_size}")

        img_data = sitk.GetArrayFromImage(img).transpose((2, 1, 0))

        min_pos = np.array([0, 0, 0])
        for i in range(0, num_splits):
            min_i = min_pos.copy()
            min_i[axis] = max(min_pos[axis], crop_size[axis] * i - overlap[axis])

            max_i = crop_size.copy()
            max_i[axis] = min(
                img_shape[axis], crop_size[axis] * (i + 1) + overlap[axis]
            )
            print(f" - Split {i} from {min_i} to {max_i}")

            img_data_i = img_data[
                min_i[0] : max_i[0], min_i[1] : max_i[1], min_i[2] : max_i[2]
            ]

            img_i = sitk.GetImageFromArray(img_data_i.transpose((2, 1, 0)))
            img_i.SetOrigin(img_origin + min_i)
            img_i.SetDirection(img_direction)
            img_i.SetSpacing(img_spacing)

            sitk.WriteImage(
                img_i,
                join(
                    output_folder,
                    image.replace(
                        "_0000.nii.gz",
                        f"__{axis}__{min_i[axis]}__{max_i[axis]}__0000.nii.gz",
                    ),
                ),
                True,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        help="Path to the folder which contains the images to be split",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Path to the folder into which the processed images should be saved in",
    )
    parser.add_argument(
        "-m",
        "--model",
        default=None,
        help="Path to the nnUNet model which will be used for predicting the split images",
    )
    parser.add_argument(
        "-s",
        "--splits",
        default=8,
        type=int,
        help="In how many splits the images should be divided to",
    )
    parser.add_argument(
        "-a",
        "--axis",
        default=2,
        type=int,
        help="In which axis the images should be splitted in",
    )
    args = parser.parse_args()

    split_files(args.input, args.output, args.model, args.splits, args.axis)
