# Welcome

This repository contains the code and documentation to run the complete nnUNet pipeline from processing, training and inference on 3d X-ray CT images. 

If you used this repository and associated code for your own work, please cite the following references: 
````
Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nature methods, 18(2), 203-211. https://doi.org/10.1038/s41592-020-01008-z

````
````
Phalempin, M., Krämer, L., Geers-Lucas, M., Isensee, F., & Schlüter, S. (2024). Advanced segmentation of soil constituents in X-ray CT images using nnUNet. Authorea Preprints. https://doi.org/10.22541/essoar.173395846.68597189/v1
````



Authors: Maxime Phalempin (UFZ) and Lars Krämer (DKFZ, HIP)

# Workflow


**Nomenclature**: The following terms are frequently used in this documentation. 
They might have some slightly different meanings in our communities, this is how I used them in this document.
- Dataset: Collection of all images and annotation
- Image: one single (.mha or .nii.gz) file which contains the grayscale values
- Annotation: one single (.mha or .nii.gz) file which contains the class ids - Created by you
- Prediction: one single (.mha or .nii.gz) file which contains the class ids - Created by the model 
# 1. Setting up your computer 
When working with Python, we often rely on various plugins and software libraries that need to be well-organized. One effective way to manage them is by using Conda environments. A Conda environment functions like a virtual workspace or isolated system, accessible through the terminal. Software installed within one Conda environment remains separate and may not be available in others. If an environment becomes unstable—for instance, due to incompatible software—you can simply create a new one and start fresh.

## 1.1. Install Miniforge 
First download and install mamba/conda. We recommend the distribution [Miniforge](https://github.com/conda-forge/miniforge#miniforge3). For ease-of-use, it is recommended to install it for your use only and to add Conda to the PATH variable during installation.

## 1.2. Install devbio-napari
Then install devbio-napari, a distribution of [Napari](https://github.com/haesleinhuepf/devbio-napari) with a set of plugins for bioimage analysis. Please use the following command in your Miniforge terminal

````
mamba create --name virtual-env python=3.11 devbio-napari pyqt -c conda-forge
````
Replace "virtual-env" by any name if you want to give to your virtual environment. Choose a name that is meaningful and easy to remember as you are likely to be using it often.  Make sure to activate your virtual environment before proceeding with further installations. In the rest of this document, we will assume that you named your virtual environment "virtual-env". 

````
mamba activate virtual-env
````

## 1.4. Install PyTorch

````
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
````

## 1.5. Install nnUNet

````
git clone https://github.com/MIC-DKFZ/nnUNet.git
cd nnUNet
pip3 install -e .
````
More information on nnUNet can be found [here](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/installation_instructions.md#installation-instructions). After installing nnUNet, make sure to set your environmental variables. 

````
set nnUNet_raw=F:\phalempin\nnUNet_raw
set nnUNet_preprocessed=F:\phalempin\nnUNet_preprocessed
set nnUNet_results=F:\phalempin\nnUNet_results
````

## 1.6. Download imageJ

Download ImageJ (Fiji) from [here](https://imagej.net/software/fiji/downloads#other-downloads).

## 1.7. Download the files from this repository and place them in their appropriate folders
1. Put the imageJ macros (files ending with .ijm) into the macros folder in the Fiji app (at .../Fiji.app/macros).
   
   **For Windows**: The files are convert_mha_to_img.ijm, convert_nii_to_mha.ijm and convert_tif_to_mha.ijm.
   
   **For Ubuntu**: convert_mha_to_img_ubuntu.ijm, convert_nii_to_mha_ubuntu.ijm.
    If the Ubuntu scripts are used, remove the _ubuntu sufices in the filenames.
3. Put the nnUNetTrainer_betterIgnoreSampling.py into nnunet/nnunetv2/training/nnUNetTrainer/variants/sampling/
4. Place nifti_io.jar into the plugins folder of ImageJ (at ../Fiji.app/plugins").
   
## 1.7.Setting file paths
In this repository, adopt \_\_path__.py: You have to define the path to your ImageJ application as well as the path to the nnUNet_raw folder (same as you set as an environment variable during the nnUNet installation).

# 2. Image annotation


# 2. How to run

You have to keep in mind that nnUNet (as well as the scripts here) are folder based. 
This means that all images should be in one folder as well as all annotations should be in one folder.
Also for predicting, all images which should be predicted have to be in one folder and the predictions will be saved into another folder.

## Training

### 2.1. Converting your data into the nnUNet format
This step takes the image and annotation files from two given folders, processes them and saves them as .nii.gz in the nnUNet_raw folder.
The processing entails:
- Transferring the .mha files to .nii.gz 
- Handling the ignore label in the annotations 
- Cropping image and annotation to the relevant parts 
- Normalizing the image crops
- Put everything into the nnUNet format (adhering to nnUNets naming conventions of folders and files and creating a dataset.json)

For each new dataset you want to use for training you have to adopt the following hyperparameters in `preprocessing_nnUNet_train.py` (starting in line 133). Some examples and additional explanations are given in the `preprocessing_nnUNet_train.py` script.

```python
input_dir_images = "" # Path to the Images
input_dir_masks = ""# Path to the Annotations
DatasetName = ""# Some Name
TaskID = 555 # A Unique ID 
Classes = ["A","B",...] # List of names for each class in the correct order
img_file_postfix = "" # Postfix of the image files, needed to find the corresponding annotation for each image
```

Afterwards, you can start converting your data by running:

````shell
python preprocessing_nnUNet_train.py
````

### 2.2. nnUNet preprocessing 
This is the default nnUNet preprocessing. This takes the data from nnUNet_raw, processes them and saves them in the nnUNet_preprocessed folder.
Depending on your data this can take a while and consume a lot of RAM.
You can run the preprocessing with the following command.
The TaskID parameter has to be the one you defined in the previous step.
The -np and -npfp parameters define how many processes are used during the preprocessing. 
A higher number means the preprocessing is faster but more RAM is consumed and with lower numbers less RAM is needed but the processing will take longer.
You can play around with this parameter, for me 4 worked well.

```shell
nnUNetv2_plan_and_preprocess -d <TaskID> -c 3d_fullres -np <num.processes> -npfp <num.processes>
# Example
nnUNetv2_plan_and_preprocess -d 555 -c 3d_fullres -np 4 -npfp 4
```

### 2.3. nnUNet training 
This step trains nnUNet with the data from nnUNet_preprocessed and saves the models, logs and checkpoints in nnUNet_results.
The content of nnUNet_preprocessed is used during training. 
If preprocessing and training are done on different devices you have to sync the nnUNet_preprocessed folder to the device on which you want to train. nnUNet is trained using 5-fold cross-validation. 
This means you have to run a separate training for each fold and each fold creates a classifier file (the checkpoints_best.pth file which contains the weights of the model). The TaskID parameter is again the one you defined in the first step.

````shell
nnUNetv2_train <TaskID> 3d_fullres <fold> -tr nnUNetTrainer_betterIgnoreSampling

nnUNetv2_train 555 3d_fullres 0 -tr nnUNetTrainer_betterIgnoreSampling
nnUNetv2_train 555 3d_fullres 1 -tr nnUNetTrainer_betterIgnoreSampling
nnUNetv2_train 555 3d_fullres 2 -tr nnUNetTrainer_betterIgnoreSampling
nnUNetv2_train 555 3d_fullres 3 -tr nnUNetTrainer_betterIgnoreSampling
nnUNetv2_train 555 3d_fullres 4 -tr nnUNetTrainer_betterIgnoreSampling
````
Note that during training, checkpoints are automatically created after 50 epochs. To resume training from a previously created checkpoint, the following command can be used.

````shell
nnUNetv2_train <TaskID> 3d_fullres <fold> -tr nnUNetTrainer_betterIgnoreSampling --c
````

### 2.4. Predicting

When the five training folds are completed we can use the model to make predictions.
For the predictions, the content of the nnUNet_results folder is needed. 
If training and predicting are done on different devices you have to sync the nnUNet_results folder to the device on which you want to predict.

1. **Preprocessing to match the nnUNet format:** Mainly convert .mha to .nii.gz, normalize the image and rename the file.
````shell
python preprocessing_nnUNet_predict.py -i <input.path.to.mha.images> -o <output.path.to.nii.gz.images>
# Example
python preprocessing_nnUNet_predict.py -i /home/l727r/Desktop/UFZ_2022_CTPoreRootSegmentation/images_mha -o /home/l727r/Desktop/UFZ_2022_CTPoreRootSegmentation/images_nii
````
2. **(Optional) Split the images:** To increase the inference speed divide the images into smaller parts to fit onto the gpu.
````shell
python preprocessing_nnUNet_predict_split.py -i <input.paht.to.imgs> -o <output.path.for.splits> -m <path.to.model> -s <num_plits>
# Also works without giving the model path, but default values are taken then
python preprocessing_nnUNet_predict_split.py -i <input.paht.to.imgs> -o <output.path.for.splits> -s <num_plits>
# Example
python preprocessing_nnUNet_predict_split.py -i /media/l727r/data/UFZ_CTPoreRootSegmentation/HI_dataset2_grass_vs_crop/test_splitting/images -o /media/l727r/data/UFZ_CTPoreRootSegmentation/HI_dataset2_grass_vs_crop/test_splitting/images_split -s 8 -m /home/l727r/Documents/cluster-checkpoints/nnUNetv2_trained_models/Dataset167_UFZ_Dataset2_grass_vs_crop_v2/nnUNetTrainer_betterIgnoreSampling_noSmooth__nnUNetPlans__3d_fullres
````
3. **nnUNet_predict:** The trained nnUNet models are used to predict the files.
````shell
nnUNetv2_predict -i <input.path.to.nii.gz.images> -o <output.path.to.nii.gz.predictions> -d <TaskID> -tr nnUNetTrainer_betterIgnoreSampling -c 3d_fullres
# Example
nnUNetv2_predict -i /home/l727r/Desktop/UFZ_2022_CTPoreRootSegmentation/images_nii -o /home/l727r/Desktop/UFZ_2022_CTPoreRootSegmentation/predictions_nii -d 555 -tr nnUNetTrainer_betterIgnoreSampling -c 3d_fullres
````
4. **(Optional) Ensemble the splitted Images:** Needs only be done if images got split in Step 2.
````shell
python postprocessing_nnUNet_predict_ensemble.py -i <input.path.to.splitted.prediction> -o <output.path.for.ensembled.predictoin>
# Example
python postprocessing_nnUNet_predict_ensemble.py -i /media/l727r/data/UFZ_CTPoreRootSegmentation/HI_dataset2_grass_vs_crop/test_splitting/splitted_prediction -o /media/l727r/data/UFZ_CTPoreRootSegmentation/HI_dataset2_grass_vs_crop/test_splitting/prediction
````

5. **Postprocessing:** Just transfer the .nii.gz files into the .mha format.
````shell
python postprocessing_nnUNet_predict.py -i <input.path.to.nii.gz.predictions> -o <output.path.to.mha.predictions>
# Example
python postprocessing_nnUNet_predict.py -i /home/l727r/Desktop/UFZ_2022_CTPoreRootSegmentation/predictions_nii -o /home/l727r/Desktop/UFZ_2022_CTPoreRootSegmentation/predictions_mha
````
# 3. How to run nnUNet on the EVE cluster
## 3.1. Short description 

The EVE cluster is a high performance computing cluster (HPC) available to the employees of the UFZ and iDiv. The cluster is equipped with eight NVIDIA Tesla A100 80G GPUs (as of 14.03.2024) which makes it a competitive environment to run GPU-based image processing jobs. The cluster relies on a Simple Linux Utility for Resource Management (SLURM) to allocate computing resources according the requirements of submitted jobs. 

## 3.2. Getting started on EVE
To get started with the EVE cluster, first contact the wkdv at the following address wkdv-cluster@ufz.de to be granted access. Once granted access, install FileZilla and X2Go Client via the portal manager. FileZilla is a software which enables file transfers between your computer and the cluster whereas X2Go Client provides a full remote graphical desktop environment to establish command-line connections to the cluster. Full instructions on how to set-up FileZilla and X2Go are provided on the [EVE Cluster wiki](https://wiki.ufz.de/eve/index.php/Main_Page).

## 3.3. Setting  up a virtual environment
It is recommended to install nnUNet and its dependencies in a python virtual environment. To do so, execute the following commmand, where "username" is your UFZ login name and "venv" is the name of your virtual environment.

````shell
python -m venv /home/username/venv
````
Then activate the virtual environment

````shell
source /home/username/venv/bin/activate
````

Then install PyTorch with the following command
````shell
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
````
Note that, here, we install an older PyTorch version (compatible with the CUDA platform 11.7). More recent versions of CUDA are currently available (i.e. 11.8 and 12.1). I was having issues with the newest version 11.8 on the EVE cluster. I have not tried the latest 12.1 version. After the successful installation of PyTorch, install nnUNet with the commands specified at the section 1.2. 

## 3.4. Storing data on EVE
There are several locations on which data can be stored on EVE. The appropriate location will depend on the filetype and their expected lifetime. It is good practice to keep all softwares, virtual environments, scripts and git repositories on the /home folder. The /work folder is suited to store the output of jobs and it has therefore a high disk quota, however the file lifetime cannot exceed 60 days. To store data which should be kept for longer period of times (i.e. trained models or grayscale data), the /data directory is approriate. We have created a directory exclusively for the use of nnUNet within the BOSYS department. This directory can be accessed under /data/bosys-nnunet/ and it currently has a disk quota of 4TB. The disk quota can be increased by the wkdv administrators upon request. 

## 3.5. Integrating the EVE cluster in the segmentation pipeline
Because of the current constraints associated with running ImageJ on a HPC, it is recommended to perform all the ImageJ and CPU-based operations on a workstation and then move the preprocessed files to the cluster to perform the GPU-based operations. This includes the conversion steps (see section 2.1) and the preprocessing steps (see section 2.2). After preprocessing, the nnUNet_preprocessed folder can be moved to the /data/bosys-nnunet directory.

## 3.6. Training nnUNet on the EVE cluster

To run the training on the EVE cluster, create a shell file (.sh) named (for instance) submit_nnunet_tr_fold0.sh using nano. 

````shell
nano submit_nnunet_tr_fold0.sh 
````
Copy then the following lines of codes within the shell file

````shell
#!/bin/bash

#SBATCH --job-name=nnunet_tr_fold0  # jobname
#SBATCH --chdir=/work/username      # sets the working directory 
#SBATCH --output=/work/%u/%x-%j.log # give name and filepath for the .log file (console output)
#SBATCH --time=1080                 # time requested for the job (in minutes)
#SBATCH --nodes=1                   # number of nodes requested
#SBATCH --ntasks=1                  # number of tasks across all nodes
#SBATCH --cpus-per-task=7           # number of cpus per tasks (>1 for multithreading). 
#SBATCH --mem-per-cpu=60G           # memory allocated per CPU. 
#SBATCH --mail-type=BEGIN,END       # request notifications upon starting and ending the job
#SBATCH -G nvidia-a100:1            # request specifically a NVIDIA A100 

###  activating  the virtual environment
source /home/username/venv/bin/activate 
## declaring the environment variable
export nnUNet_preprocessed="/data/bosys-nnunet/nnUNet_preprocessed" 
export nnUNet_results="/work/phalempi/nnUNet_results" 

## Set the number of processes used for data augmentation PER GPU
export nnUNet_n_proc_DA=28 # the faster the GPU, the higher the value

## run nnUNet training command (see section 2.3)
nnUNetv2_train 444 3d_fullres 0 -tr nnUNetTrainer_betterIgnoreSampling
````
Then submit the shell script as a sequential job using the following command. 

````shell
sbatch submit_nnunet_tr_fold0.sh 
````
You can repeat the operations and create several shell scripts (i.e. one for each training fold) and submit each job one after the other. Depending on the resources currently available, the SLURM system will distribute each training fold to a GPU, so that all the five training will run simultaneously. 

Please consider the following when requesting resources on the EVE cluster. One of the most important resource at EVE is the maximum runtime of jobs. It specifies a limit which a running job may not exceed. If the job exceeds the requested time, it will be killed automatically by the scheduler. The same applies for the requested memory per cpu. It is a good practice to optimize these parameters to avoid exceeding the job requirements, but to keep them as low as possible so that the scheduler grants resources quicker. Note also that GPU Nodes only have a maximum of 470GB RAM available. This means that the total amount of RAM (calculated as cpus-per-task * mem-per-cpu) has to be inferior to 470GB. 

## 3.7. Preparing of array jobs for the predictions
In order to run the predictions in a parallelized fashion on the EVE cluster, each image has to be placed in its specific folder. The reason is that nnUNet functions in a folder based manner, i.e., it predicts all the images present in given folder. However, if all the images to be predicted were in the same folder, the memory and time requirements would be huge, and the scheduler would not allocate resouces to the job. By putting one image in a single folder, the memory and time requirements are low and the images can be distributed across all nodes hosting GPUs. To spare you the effort of moving each image into an individual folders, we have created a shell script that does the job for you. The shell script takes two arguments: (1) the input folder containing the images to be predicted and (2) the directory path where the folders will each individual folders will be created. Those arguments are given after the flags -i and -o, respectively. You can then run the shell script with the following command. 

````shell
sh mkdir_movefiles.sh -i /path/to/input/images -o /path/to/output/images
````
Note that if "/path/to/output/image" does not exist, it will be created automatically. Note also that here, we do not need to submit the job to the scheduler but can simply run it with a "sh" command. After running this shell script, all images are now placed into a single individual folder and are ready to be processed.

## 3.8. Running predictions with an array job
In order to run the predictions in a parallelized manner, we have to create a so-called "array job". Job arrays allow to use SLURM's ability to create multiple jobs from one script. For example, instead of having 5 submission scripts to run the same job step with different arguments, we can have one script to run the 5 job steps at once. This allows to leverage EVE´s ability to process images simulateneously (x GPUs process x images at the same time). To do so, prepare a shell script (named for instance submit_nnUnet_array_list.sh) and copy the following commands in it.

````shell
#!/bin/bash

#SBATCH --job-name=nnunet_prediction
#SBATCH --chdir=/work/phalempi
#SBATCH --output=/work/%u/%x-%A-%a.log
#SBATCH --time=45
#SBATCH --mem-per-cpu=60G
#SBATCH --mail-type=BEGIN,END
#SBATCH -G nvidia-a100:1

source  /home/phalempi/MPvenv/bin/activate ##  activating  the virtual environment
export nnUNet_results="/work/phalempi/nnUNet_results" ## declaring the environment variable

# Retrieving the name of each sample, storing it in a list and using the array job submission to submit all jobs at once
# Define the folder path
folder_path="/work/phalempi/grayscale_data"

# Initialize an empty list
file_list=()

# Loop through all files in the folder and add them to the list
for file in "$folder_path"/*; do
    # Check if it's a file (not a directory)
    if [ -d "$file" ]; then
        file_list+=("$(basename "$file")")
    fi
done

nnUNetv2_predict -i "$folder_path"/"${file_list[$SLURM_ARRAY_TASK_ID]}" -o /work/phalempi/predictions/images_nii -d 444 -tr nnUNetTrainer_betterIgnoreSampling -c 3d_fullres
# here the number of task id for the scheduler as an index to retrieve the name of the images to be predicted (stored in a list called file_list)
````
To submit the array job, use the following command

````shell
sbatch –a 0-412 submit_nnUnet_array_list.sh  
````
where the parameters "0" and "412" are used to initiate and terminate the "for" loop with which the name of all the samples in the dataset are retrieved. "0" is fixed and "412" is the total number of images to predict (to be modified). Depending on the resources available, the SLURM system will distribute each image to be predicted to a GPU. To check the status of your submitted jobs, you can enter the following command in your terminal.

````shell
sacct
````
To see all available commands related to job monitoring, consult the following [link](https://wiki.ufz.de/eve/index.php/Job_Monitoring_SLURM).Additionnaly, you can track the GPU activity over time with [Grafana](https://grafana.web-intern.app.ufz.de) dashboard.

# Comments
- **Annotation Process:** For annotation we used the following protocol which works well and we recommend to stay with it for future datasets:
  - The value 0 in the annotations means not labeled
  - Completely annotate one slice in the image.
  - If possible select slices from as many different images as possible (best case only 1 slice from a sinlge image).
  - Add some annotations near the annotated slice and label underrepresented or unrepresentative classes (each class should be present)
- **Class IDs**: In your Annotations class id 0 has the meaning of not beeing annotated. 
In the prediction this is not needed since we have dense predictions, this means each class id is reduced by 1 (class id annotation == class id prediction +1)
- **Number of Annotations:** With the current setup at least 5 annotations are needed (basic behaviour of nnUNet). 
You can also train with fewer annotations but this need some additional adaptation.
I recommend to not use less than 5 annotations but if you think the scenario will occur let us know and we can adopt the scripts (only minor changes, no worry).
- **Runtime reduction:** There is some tradeoff between runtime and performance, the current setup aims for getting the best result. 
If you want to reduce the runtime, there are different ways to archive this, but with the downside of loosing some performance.
Again, just let us know if you want same changes here.

# Acknowledgements
This repository was developed in the framework of a collaboration between the Department of Soil System Sciences of the [Helmholtz Center for Environmental Research](https://www.ufz.de/) and the Applied Computer Vision Lab of [Helmholtz Imaging](https://www.helmholtz-imaging.de/). Part of this work was funded by Helmholtz Imaging (HI), a platform of the Helmholtz Incubator. 

<p align="left">
  <img src="Figures/HI_Logo.png" width="150"> &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="Figures/DKFZ_Logo.png" width="500"> 
</p>
