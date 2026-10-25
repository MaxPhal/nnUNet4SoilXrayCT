import argparse
import os
import sys
from os.path import join
from pathlib import Path

import numpy as np
import SimpleITK as sitk

SPLIT_SUFFIX = "_.nii.gz"  # nnUNet strips "_0000", leaving "..._.nii.gz"

def find_split_files(img_folder: str, image_root: str) -> list:
    """
    Find all predicted split files belonging to a single image.

    Split files are named {image_root}__{axis}__{min}__{max}_.nii.gz. The root
    must match exactly, so "case1" never picks up files of "case10".

    :return: list of (axis, min_pos, max_pos, path), sorted by min_pos
    """
    splits = []
    for name in os.listdir(img_folder):
        if not name.endswith(SPLIT_SUFFIX):
            continue
        parts = name[: -len(SPLIT_SUFFIX)].rsplit("__", 3)
        if len(parts) != 4 or parts[0] != image_root:
            continue
        try:
            axis, min_pos, max_pos = (int(p) for p in parts[1:])
        except ValueError:
            continue
        splits.append((axis, min_pos, max_pos, join(img_folder, name)))

    return sorted(splits, key=lambda s: s[1])


def ensemble_image(img_folder: str, output_folder: str, image_root: str) -> None:
    """
    Concatenate all predicted splits of ONE image back into a single volume.

    :param img_folder: folder containing the predicted split files
    :param output_folder: folder the reassembled image is saved to
    :param image_root: name of the image (split filenames without the
        __axis__min__max_ part)
    """
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    splits = find_split_files(img_folder, image_root)
    if not splits:
        raise FileNotFoundError(f"No split files found for '{image_root}' in {img_folder}")
    print(f"{image_root}: {len(splits)} files found")

    axes = {s[0] for s in splits}
    if len(axes) != 1:
        raise ValueError(f"Splits of '{image_root}' use different axes: {sorted(axes)}")
    axis = axes.pop()

    min_pos = np.array([s[1] for s in splits])
    max_pos = np.array([s[2] for s in splits])
    paths = [s[3] for s in splits]

    overlap = (
        np.insert(max_pos, 0, min_pos[0])
        - np.insert(min_pos, len(min_pos), max_pos[-1])
    ) / 2

    for i, path in enumerate(paths):
        img_i = sitk.ReadImage(path)
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
    """
    Reassembles the predicted splits of a single image. Meant to be called
    once per image, e.g. once per SLURM array task, with the task -> image
    root mapping handled in the submission script (bash)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Path to the folder which contains the predicted split files",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Path to the folder into which the reassembled image should be saved "
        "(may be shared across parallel jobs)",
    )
    parser.add_argument(
        "-r",
        "--root",
        required=True,
        help="Name of the image to reassemble (split filenames without the "
        "__axis__min__max_.nii.gz part)",
    )

    args = parser.parse_args()

    slurm_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if slurm_task_id is not None:
        print(f"SLURM_ARRAY_TASK_ID: {slurm_task_id}")

    try:
        ensemble_image(args.input, args.output, args.root)
        print("\n----------\nFinished successfully.")
    except Exception as e:
        print(f"\n----------\n[FAILED] {args.root}: {e}")
        # Exit non-zero so SLURM records this task as failed
        sys.exit(1)