import argparse
import json
import os
import sys
from os.path import join
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import tifffile

from prepare_raw_data_for_training_hpc import img_normalize, read_metadata

# ---------------------------------------------------------------------------
# Step 2/3: read a (multi-page / 3D) .tif directly in Python, no ImageJ needed
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


def read_tif_as_sitk(tif_path: str) -> sitk.Image:
    """
    Read a .tif image (2D or 3D/multi-page) via tifffile and wrap it as a
    SimpleITK image, carrying over the embedded resolution/spacing metadata.

    tifffile returns array axes as (Z, Y, X) for a 3D stack, which matches
    what sitk.GetImageFromArray expects. Origin and direction aren't stored
    in a plain .tif, so they stay at SimpleITK's defaults (zero origin,
    identity direction).
    """
    with tifffile.TiffFile(tif_path) as tif:
        arr = tif.asarray()
        x_spacing, y_spacing, z_spacing = extract_spacing_from_tif(tif)

    img = sitk.GetImageFromArray(arr)
    if img.GetDimension() == 3:
        img.SetSpacing((x_spacing, y_spacing, z_spacing))
    else:
        img.SetSpacing((x_spacing, y_spacing))

    return img

# ---------------------------------------------------------------------------
# Steps 4/5: split (or copy) the opened image and save into its own folder
# ---------------------------------------------------------------------------
def normalize_image(img: sitk.Image, norm_type: str) -> sitk.Image:
    """
    Normalize a SimpleITK image via img_normalize, preserving its spacing/
    origin/direction. Returns img unchanged if norm_type is "noNorm".
    """
    if norm_type == "noNorm":
        return img

    img_arr = sitk.GetArrayFromImage(img).astype(np.uint8)
    img_arr = img_normalize(img_arr, norm_type)
    normalized = sitk.GetImageFromArray(img_arr)
    normalized.CopyInformation(img)
    return normalized


def copy_and_save(img: sitk.Image, case_name: str, case_folder: str, norm_type: str) -> None:
    """
    Step 4 (splits == 0) / Step 5: optionally normalize the whole image, then
    save it unsplit into its own case folder.
    """
    img = normalize_image(img, norm_type)
    sitk.WriteImage(img, join(case_folder, f"{case_name}_0000.nii.gz"), True)

def split_and_save(
    img: sitk.Image,
    case_name: str,
    output_dir: str,
    norm_type: str,
    num_splits: int,
    axis: int,
    patch_size,
    target_spacing,
) -> None:
    """
    Step 4 (splits > 0) / Step 5: optionally normalize the whole image, split
    it into overlapping patches along `axis`, and save each patch into its
    own folder directly under output_dir (one folder per split, named after
    case/axis/min/max, e.g. for distributing across GPUs).
    """
    img = normalize_image(img, norm_type)

    img_shape = img.GetSize()
    img_spacing = img.GetSpacing()
    img_direction = img.GetDirection()
    img_origin = img.GetOrigin()

    print(f"Image Shape:   {img_shape}")
    print(f"Image Spacing: {img_spacing}")

    original_patch_size = patch_size * np.array(target_spacing) / np.array(img_spacing)
    overlap = np.ceil(original_patch_size / 2)
    crop_size = np.array(img_shape)
    crop_size[axis] = np.ceil(img_shape[axis] / num_splits)

    print(f"Original Patch Size:   {original_patch_size}")
    print(f"Patch Overlap:   {overlap}")
    print(f"Base Crop Size:   {crop_size}")

    img_data = sitk.GetArrayFromImage(img).transpose((2, 1, 0))

    min_pos = np.array([0, 0, 0])
    for i in range(num_splits):
        min_i = min_pos.copy()
        min_i[axis] = max(min_pos[axis], crop_size[axis] * i - overlap[axis])

        max_i = crop_size.copy()
        max_i[axis] = min(img_shape[axis], crop_size[axis] * (i + 1) + overlap[axis])
        print(f" - Split {i} from {min_i} to {max_i}")

        img_data_i = img_data[min_i[0]:max_i[0], min_i[1]:max_i[1], min_i[2]:max_i[2]]

        img_i = sitk.GetImageFromArray(img_data_i.transpose((2, 1, 0)))
        img_i.SetOrigin(img_origin + min_i)
        img_i.SetDirection(img_direction)
        img_i.SetSpacing(img_spacing)

        # Each split gets its own folder, named after case/axis/min/max only
        # (no split index — min/max alone indicate position in the stack)
        split_name = f"{case_name}__{axis}__{min_i[axis]}__{max_i[axis]}"
        split_folder = join(output_dir, split_name)
        Path(split_folder).mkdir(parents=True, exist_ok=True)

        sitk.WriteImage(
            img_i,
            join(split_folder, f"{split_name}__0000.nii.gz"),
            True,
        )


# ---------------------------------------------------------------------------
# Per-image pipeline (steps 1-5)
# ---------------------------------------------------------------------------
def process_image(
    tif_path: str,
    output_dir: str,
    norm_type: str,
    num_splits: int,
    axis: int,
    patch_size,
    target_spacing,
) -> None:
    """
    Run the full pipeline on a single image:
      1. Read the image name
      2. Read the .tif directly in Python (tifffile) and wrap as a SimpleITK image
      3. Optionally normalize (if norm_type != "noNorm"), then split it if num_splits > 0, otherwise save it unsplit
      4. Save the result(s) into their own folder under output_dir

    """
    # Step 1
    case_name = os.path.splitext(os.path.basename(tif_path))[0]
    print("----------")
    print(f"Image File:    {case_name}")

    # Step 2 + 3: read the .tif
    img = read_tif_as_sitk(tif_path)

    # Step 4 + 5
    if num_splits > 0:
        # Each split gets its own folder directly under output_dir, named
        # after case/axis/min/max so it's self-describing without needing
        # a parent case_name folder
        split_and_save(
            img, case_name, output_dir, norm_type, num_splits, axis, patch_size, target_spacing
        )
    else:
        case_folder = join(output_dir, case_name)
        Path(case_folder).mkdir(parents=True, exist_ok=True)
        copy_and_save(img, case_name, case_folder, norm_type)

if __name__ == "__main__":
    """
    Runs the pipeline on a single .tif image, passed in via --input. Meant to
    be called once per image, e.g. once per SLURM array task, with the task
    -> filename mapping handled in the submission script (bash)

    Steps:
      1. Read the image's name
      2. Read the .tif directly in Python and wrap as a SimpleITK image
      3. Optionally normalize (--norm != noNorm), then split it (--splits > 0) or save it unsplit (--splits 0)
      4. Save the result into its own folder under --output
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        required= True,
        help="Path to a single input .tif image to process",
    )
    parser.add_argument(
        "-o",
        "--output",
        required= True,
        help="Path to the folder into which the final image(s) should be saved "
        "(may be shared across parallel jobs - each image/split gets its own "
        "subfolder)",
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
        help="In how many splits the image should be divided. Use 0 to skip "
        "splitting entirely and copy the image through as-is",
    )
    parser.add_argument(
        "-a",
        "--axis",
        default=2,
        type=int,
        help="In which axis the image should be split",
    )
    args = parser.parse_args()

    # Extract metadata information from .json file
    metadata = read_metadata(Path.cwd() / 'dataset_info.json')  
    norm_type = metadata["norm_type"] # this is the normalization type

    # create output dir if not exists
    Path(args.output).mkdir(parents=True, exist_ok=True)

    # get slurm task ID from OS.env
    slurm_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if slurm_task_id is not None:
        print(f"SLURM_ARRAY_TASK_ID: {slurm_task_id}")

    # retrieve patch size and target spacing
    if args.model is not None:
        with open(join(args.model, "plans.json"), "r") as file:
            plans = json.load(file)
        patch_size = plans["configurations"]["3d_fullres"]["patch_size"][::-1]
        target_spacing = plans["configurations"]["3d_fullres"]["spacing"][::-1]
    else:
        print(
            "Warning: No model_path was given, default patch_size and target_spacing "
            "are taken, for exact values give the model path"
        )
        patch_size = [224, 224, 48]
        target_spacing = [1.0, 1.0, 1.0]

    print(f"Model Parameters:")
    print(f"Patch Size:     {patch_size}")
    print(f"Target Spacing: {target_spacing}")

    try:
        process_image(
            args.input,
            args.output,
            norm_type,
            args.splits,
            args.axis,
            patch_size,
            target_spacing,
        )
        print("\n----------\nFinished successfully.")
    except Exception as e:
        print(f"\n----------\n[FAILED] {args.input}: {e}")
        # Exit non-zero so SLURM (or any job scheduler) records this task as failed
        sys.exit(1)