import argparse
import os
from os.path import join

import numpy as np
import SimpleITK as sitk

def ensemble_files(img_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    images = os.listdir(img_folder)
    images = [image for image in images if image.endswith(".nii.gz")]
    image_roots = np.unique([img.rsplit("__", 4)[0] for img in images])

    for image_root in image_roots:
        images_i = [image for image in images if image_root in image]

        print(f"{image_root}: {len(images_i)} files found")

        params = np.array(
            # [image.replace("__0000.nii.gz", "").rsplit("__", 3)[-3:] for image in images_i],
            [image.replace("_.nii.gz", "").rsplit("__", 3)[-3:] for image in images_i],
            dtype=int,
        )

        axis = params[0, 0]
        min_pos = np.sort(params[:, 1])
        max_pos = np.sort(params[:, 2])
        overlap = (
            np.insert(max_pos, 0, min_pos[0])
            - np.insert(min_pos, len(min_pos), max_pos[-1])
        ) / 2

        for i in range(0, len(images_i)):
            # file = f"{image_root}__{axis}__{min_pos[i]}__{max_pos[i]}__0000.nii.gz"
            file = f"{image_root}__{axis}__{min_pos[i]}__{max_pos[i]}_.nii.gz"
            img_i = sitk.ReadImage(join(img_folder, file))

            img_i_data = sitk.GetArrayFromImage(img_i).transpose((2, 1, 0))

            min_i = np.array([0, 0, 0])
            min_i[axis] = min_i[axis] + np.floor(overlap[i])

            max_i = np.array(img_i.GetSize())
            max_i[axis] = max_i[axis] - np.floor(overlap[i + 1])

            img_i_data = img_i_data[
                min_i[0] : max_i[0], min_i[1] : max_i[1], min_i[2] : max_i[2]
            ]

            if i == 0:
                img_data = img_i_data
                img_spacing = img_i.GetSpacing()
                img_direction = img_i.GetDirection()
                img_origin = img_i.GetOrigin()
            else:
                img_data = np.concatenate((img_data, img_i_data), axis=axis)

        img = sitk.GetImageFromArray(img_data.transpose((2, 1, 0)))

        img.SetOrigin(img_origin)
        img.SetDirection(img_direction)
        img.SetSpacing(img_spacing)

        sitk.WriteImage(
            img, join(output_folder, image_root + ".nii.gz"), useCompression=True
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
    args = parser.parse_args()

    ensemble_files(args.input, args.output)
