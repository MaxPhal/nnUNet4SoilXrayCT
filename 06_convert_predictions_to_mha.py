import argparse
import subprocess
from pathlib import Path

from __path__ import PATH_ImageJ


def nii_to_mha(input_dir: str, output_dir: str) -> None:
    """
    Use ImageJ and the convert_nii_to_mha macro script to convert .nii.gz files to .mha files

    :param input_dir: path to the input folder which contains the .nii.gz files
    :param output_dir: path to the folder in which the output should be saved in
    :return:
    """
    # os.makedirs(output_dir, exist_ok=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print("Process .nii.gz to .mha")
    subprocess.Popen(
        rf'{PATH_ImageJ} --headless -macro convert_nii_to_mha "{input_dir}--{output_dir}"',
        # fr'{PATH_ImageJ} -macro convert_nii_to_mha "{input_dir}--{output_dir}"',
        shell=True,
    ).wait()


if __name__ == "__main__":
    """
    This script convert each prediction file in the input folder from .nii.gz to .mha and saves it
    to the the output folder
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        help="Path to the folder which contains the predictions in the .nii.gz file format",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Path to the folder into which the predictions in .mha format should be saved in",
    )
    args = parser.parse_args()

    input_dir_preds = args.input
    output_dir_preds = args.output
    nii_to_mha(input_dir_preds, output_dir_preds)
