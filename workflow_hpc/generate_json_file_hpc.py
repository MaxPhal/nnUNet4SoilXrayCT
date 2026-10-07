import glob
import os
from pathlib import Path
from os.path import join
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from prepare_raw_data_for_training_hpc import read_metadata
from __path__ import PATH_nnUNet_raw

if __name__ == "__main__":
    """
    Create the dataset.json which is needed for nnUNet and contains information about the dataset
    """   
    # Read metadata 
    metadata = read_metadata(Path.cwd() / 'dataset_info.json')  

    # Extract metadata information from .json file
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
    output_folder = join(PATH_nnUNet_raw, f"Dataset{TaskID}_{DatasetName}")

    """
    Manage Folders
    """
    nnUNet_img_folder = join(output_folder, "imagesTr")

    """
    Step4: Create the dataset.json which is needed for nnUNet and contains information about the dataset
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
    print(f"Dataset metadata JSON file was printed at: {output_folder}")
