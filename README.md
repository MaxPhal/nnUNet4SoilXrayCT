# Welcome
This repository contains the code and documentation to run the complete nnUNet pipeline on 3D X-ray CT images of soil samples. It was developed in the framework of a collaboration between the Department of Soil System Sciences of the [Helmholtz Center for Environmental Research](https://www.ufz.de/) and the Applied Computer Vision Lab of [Helmholtz Imaging](https://www.helmholtz-imaging.de/). The main purpose of the collaboration was to promote and ease the adoption of deep learning for image segmentation tasks in soil science, with a focus on 3D X-ray CT images. Our contribution is three-fold, i.e., we developed:  
1. A new methodology to provide an efficient labeling strategy to obtain ground-truth annotations.
2. Several scripts to prepare the input images to a format that is compatible with nnUNet.
3. Extra utilities to extract results in a format that can be diretly used for generating figures for scientific papers. These results include data such as Dice scores or the value of the Loss function as a function of the number of epochs during training. 

In this repository, we provide detailed explanations on how to transition from a 3D image stack to nnUNet predictions. The repository was written asumming (almost) no prerequisite programming experience of the user. In this way, we hope that it can reach a broader audience. If you used this repository and associated code for your own work, please cite the following references to acknowledge our efforts: 
````
Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nature methods, 18(2), 203-211. https://doi.org/10.1038/s41592-020-01008-z
````
````
Phalempin, M., Krämer, L., Geers-Lucas, M., Isensee, F., & Schlüter, S. (2024). Advanced segmentation of soil constituents in X-ray CT images using nnUNet. Authorea Preprints. https://doi.org/10.22541/essoar.173395846.68597189/v1
````
We hope you will find this repository helpful! We wish a you a lot of fun working with nnUNet :). Feel free to contact us if you want to share your experience using nnUNet on your X-ray CT images of soil samples.

This repository was drafted by Maxime Phalempin (UFZ) and Lars Krämer (DKFZ, HIP). It was reviewed and edited by Steffen Schlüter (UFZ), Maik Geers-Lucas (TUBerlin) and Fabian Isensee (DKFZ, HIP),

# Nomenclature
Before getting down to business, let´s define a few terms to avoid any confusion. 
- Dataset: Collection of all images and annotation
- Image: one single (.tif, .mha or .nii.gz) file which contains the grayscale values
- Class: a category present in the images, e.g., for instance "roots", "soil matrix" or "biopores" 
- Label: the specific value assigned in the segmentation mask, e.g., 1 = Roots; 2= Soil matrix; 3 = Biopores
- Annotation: one single (.tif, .mha or .nii.gz) file which contains the labelled classes - Created by you
- Prediction: one single (.tif, .mha or .nii.gz) file which contains the labelled classes - Created by nnUNet
- Mask: a separate image that defines which pixels belong to specific classes (or labels). Masks can be binary if only background and foregound are of intestest or grayscale for multiclass segmentation. Essentially, an annotation file is also a mask and both words can be used interchangeably in this documentation.  
    
# Workflow
Our workflow includes several crucial steps such as image annotation, conversion, preprocessing, model training, inference and analysis of the output data (Figure 1). The workflow was mainly developed in a Python environment. It uses several scripts (steps in italic font on figure 1) which create annotations and convert the images to nnUNet-friendly formats, before processing using the native nnUNet pipeline (steps in bold font on figure 1). 

<p align="center">
  <img src="Figures/Workflow.png" width="500"> 
</p>

**Figure 1.** Workflow to transition from 3D X-ray CT image stacks to nnUNet predictions 

Our workflow relies on the use of high performance computing (HPC) cluster to perform computionnally demanding tasks such as training and inference. We highly recommend you do the same because GPUs are so much faster than CPUs. Also, we have developed this workflow in a way that several GPUs can work in parallel on several cutouts of the same image. This feature allows increased processing speed, which makes it highly competitive, even against less demanding segmentation methods. If your university or research insitution does not offer access to a HPC cluster for scientific computation, consider relying on dedicated GPU Servers that can be rented. For processing tasks that rely on CPUs only, we recommend using on a regular workstation. To develop the workflow, we used a workstation running on Windows (64-bit, 767 GB RAM) for CPU tasks only. For GPU tasks only, we used the [EVE cluster](https://www.ufz.de/index.php?en=51499) of the UFZ. 

# 1.  Groundtruth data preparation 
# 1.1 Setting up your computer 
## 1.1.1 Install Miniforge 
Miniforge is a lightweight version of Anaconda that helps you install and manage Python and other software packages efficiently. It’s designed for flexibility and supports open-source package management with Conda, making it ideal for scientific computing and data analysis. We recommend the distribution [Miniforge](https://github.com/conda-forge/miniforge#miniforge3). For ease-of-use, it is recommended to install it for your use only and to add Conda to the PATH variable during installation.

## 1.1.2. Create a virtual environment <!-- Successful on BOPHY116 -->
When working with Python, we often rely on various plugins and software libraries that need to be well-organized. One effective way to manage them is by using Conda environments. A Conda environment functions like a virtual workspace or isolated system, accessible through the terminal. Software installed within one Conda environment remains separate and may not be available in others. If an environment becomes unstable—for instance, due to incompatible software—you can simply create a new one and start fresh. Please use the following command in your Miniforge terminal to create a virtual environment.
````
mamba create -n venv-napari python=3.11
````
Replace "venv-napari" by any name if you want to give to your virtual environment. Choose a name that is meaningful and easy to remember as you are likely to be using it often.  Make sure to activate your virtual environment before proceeding with further installations. Once the virtual environment is created, activate it with: 
````
mamba activate venv-napari
````
## 1.1.3. Install devbio-napari <!-- Successful on BOPHY116 -->
Napari is an open-source tool for viewing and analyzing large 2D and 3D images, commonly used in scientific research. It provides an interactive, user-friendly interface for exploring image data, making annotations, and applying analysis techniques. What we love so much about Napari is that it is scriptable which makes it really easy to work with. We recommend devbio-napari, a distribution of [Napari](https://github.com/haesleinhuepf/devbio-napari) with a set of plugins for bioimage analysis. In our workflow, we used Napari to annotate images. Please use the following command in your Miniforge terminal to install devbio-napari.
````
mamba install devbio-napari pyqt -c conda-forge
````

# 1.2. Image selection
For the annotations, make sure to select slices from as many different images as possible (best case only 1 slice from a single image) and to add some annotations near the annotated slice and label underrepresented or unrepresentative classes. In the annotations, the label 0 should never be annotated as it is the class "To be predicted". We recommend to prepare image cutouts having the same x-y extent as the original but only 300 slices in the Z-direction. 
IMPORTANT: The selected images for groundtruth generation should be in a 3D .tif file format.  

# 1.3. Image annotation
For image annotation, we developed a strategy that minimizes annotation efforts while still ensuring that all relevant classes are captured. This strategy relies on dense annotations of one slice and on sparse annotations for interesting features within a stack (see figure 2 in our [publication](https://doi.org/10.22541/essoar.173395846.68597189/v1)). To perform dense annotations, the middle slice of the stack was segmented with Otsu thresholding and heavily annotated manually for other interesting features. To do so in a semi-automatic manner, we created the `make_annotations.py` script. Before launching this script in your Miniforge terminal, make sure to adapt the lines 60 to 86 according to the classes that you want to segment. Below, we show the class definition that we used for the Dataset 1 (see our [publication](https://doi.org/10.22541/essoar.173395846.68597189/v1) for more information).

````
label_names = {
    0: "ToPredict",
    1: "Matrix",
    2: "Wall",
    3: "Rocks",
    4: "FreshRoots",
    5: "DegradingRoots",
    6: "otherPOM",
    7: "RootChannels",
    8: "EarthwormBurrows",
    9: "otherPores",
}

# Define a colormap for labels
color_dict = {
    0: (  0,  0,   0),  # ToPredict
    1: (115,  0, 102),  # Matrix
    2: ( 51, 51,  51),  # Wall
    3: (255,255,   0),  # Rocks
    4: (204,  0,   0),  # FreshRoots
    5: (255,153,  51),  # DegradingRoots
    6: (204,255, 255),  # otherPOM
    7: ( 11,  0, 255),  # RootChannels
    8: ( 51,204,   0),  # EarthwormBurrows
    9: (  0,122, 153),  # otherPores
}
````
Note that the label 1 should also be set to the soil matrix. Once the classes are modified according to your dataset and the colors are defined for each label, you can launch the `make_annotations.py` script. This script takes three arguments, i.e., the input folder path (the path to the folder containing the images that you want to annotate), the output folder path (the path to the folder where the annotations will be saved) and the sample ID (the name of your image). These arguments are passed from the command terminal with the three flags "-i", "-o" and "-id" respectively.   
````
python make_annotations.py -i </path/to/the/images/to/annotate> -o </path/to/where/annotations/will_be/saved> -id <sample_ID>
# Example
python make_annotations.py -i C:\Users\phalempi\Desktop\images -o C:\Users\phalempi\Desktop\annotations -id SPP_P21_SPE_UC193
````
Just a few moment after launching the script, the image name and shape will be printed in the terminal. Afterwards, the GUI of Napari pops up and displays the middle slice of the loaded image. On that middle slice, the soil matrix appears in color (115,  0, 102) (RGB). To get familiar with the GUI of Napari, we recommend to consult external resources. There are some very good explanatory videos out there (for instance on YouTube) that show how to efficiently annotate images with Napari. Because a short video is more impactful than a those words, we won´t go over the procedure in this repository. Note that with the current version of nnUNet at least five annotations are needed. In all cases, we recommend not using than five annotations.

Once you are done annotating your images, just close the Napari GUI and the annotated images will be automatically saved under the path you have given after the flag -o. Before going further with data preparation, make sure to deactivate the current virtual environment.
````
mamba deactivate
````
# 2. Data preparation 
# 2.1. Setting up your computer 
## 2.1.1. Create a virtual environment <!-- Successful on BOPHY116 -->
Unfortunately, devbio-napari can not be installed in the same virtual environment because of conflicts between version packages. This means that we are gonna need to install another virtual environment to work with nnUNet. We can create a virtual environment and activate with a single command line as follows: 
````
mamba create -n venv-nnunet python=3.11 && conda activate venv-nnunet
````
## 2.1.2. Install git <!-- Successful on BOPHY116 -->
Git is a version control system that helps track changes in code and collaborate on projects. Git can also be used to download code as it allows users to clone repositories from remote servers (e.g., GitHub, GitLab, Bitbucket). This is useful for accessing open-source projects, collaborating on code, or deploying software. Here, we are gonna to use git to download the nnUNet repository from GitHub. To install git, type the following command in your terminal.
````
mamba install git
````
## 2.1.3. Install nnUNet <!-- Successful on BOPHY116 -->
nnUNet is a self-configuring deep learning framework for medical image segmentation. It automatically adapts to new datasets by optimizing preprocessing, network architecture, and training settings, making it a powerful and user-friendly tool for segmentation tasks. More information on nnUNet can be found [here](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/installation_instructions.md#installation-instructions). To install nnUNet, enter the following commands: 
````
git clone https://github.com/MIC-DKFZ/nnUNet.git
cd nnUNet
pip3 install -e .
````
After installing nnUNet, create some folders named "nnUNet_raw" and "nnUNet_preprocessed". After creating the folders, set them as environmental variables using the following commands in your Miniforge
terminal: 
````
set nnUNet_raw= your/path/to/nnUNet_raw
set nnUNet_preprocessed=your/path/to/nnUNet_preprocessed
````

## 2.1.4. Download ImageJ <!-- Successful on BOPHY116 -->
ImageJ is a free, open-source image processing software widely used in scientific research. In our workflow, we used ImageJ to convert the input images to a nnUNet-friendly format. You can download ImageJ (Fiji) from [here](https://imagej.net/software/fiji/downloads#other-downloads).

## 2.1.5. Download the files from this repository and place them in appropriate folders 
1. Put the imageJ macros (files ending with .ijm) into the macros folder in the Fiji app (at .../Fiji.app/macros).
   
   **For Windows**: The files are convert_mha_to_img.ijm, convert_nii_to_mha.ijm and convert_tif_to_mha.ijm.
   
   **For Ubuntu**: convert_mha_to_img_ubuntu.ijm, convert_nii_to_mha_ubuntu.ijm.
    If the Ubuntu scripts are used, remove the _ubuntu sufices in the filenames.
3. Put the nnUNetTrainer_betterIgnoreSampling.py into nnunet/nnunetv2/training/nnUNetTrainer/variants/sampling/
4. Place nifti_io.jar into the plugins folder of ImageJ (at ../Fiji.app/plugins").
   
## 2.1.6. Setting file paths
Open the \_\_path__.py file (from this repository) with a Text Editor and adapt the paths according to your local installations. You have to define the four following paths: 
1) path to your ImageJ application
2) the path to the nnUNet_raw folder (same as you set as an environment variable during the nnUNet installation)
3) the path to the images of your training data (input_dir_images) # IMPORTANT: images should be in 3D stack .tif format
4) the path to your annotations (input_dir_masks) # IMPORTANT: annotations should already be in 3D stack .tif format if you used `make_annotations.py` to generate the annotations

# 2.2. Data conversion
This step takes the image and annotation files from two given folders, processes them and saves them as .nii.gz in the nnUNet_raw folder. Here, you have to keep in mind that our workflow was developed so as to work in a "folder-based" manner. 
This means that all images should be in one folder and all annotations should be in another one. Data conversion entails converting the input files to .nii.gz, handling the ignore label in the annotations, cropping image and annotation to the relevant parts, normalizing the image crops and putting everything into the nnUNet format (adhering to nnUNets naming conventions of folders and files and creating a dataset.json). For each new dataset, you have to adapt the following hyperparameters in `preprocessing_nnUNet_train.py` (starting in line 133). Some examples and additional explanations are given in the `preprocessing_nnUNet_train.py` script.

```python
DatasetName = ""# Some Name
TaskID = 555 # A Unique ID 
Classes = ["A","B",...] # List of names for each class in the correct order
norm_type= "zscore" # one of [noNorm, zscore, rescale_to_0_1, rgb_to_0_1] with default==zscore
img_file_postfix = "" # Postfix of the image files, needed to find the corresponding annotation for each image
```
**Note:** You have the following options for the normalization. It is important to select the same normalization type when running preprocessing_nnUNet_predict:
- *noNorm*: No normalization is done
- *zscore*: This is the default, normalize by mean and std: ```(img - mean_) / std_```
- *rescale_to_0_1*: rescale the values to the range of 0 and 1: ```(img-min_)/(max_-min_)```
- *rgb_to_0_1*: Just divide by 255: ```img/255```

Once you have modified the hyperparameters accordingly, you can start converting your data by running:
````shell
python preprocessing_nnUNet_train.py
````

# 2.3. Preprocessing and planing
The preprocessing and experiment planing are default steps in the native nnUNet pipeline. These steps use the data from nnUNet_raw, processes them and saves them in the nnUNet_preprocessed folder. Depending on your data this can take a while and consume a lot of RAM. You can run the preprocessing with the following command.

```shell
nnUNetv2_plan_and_preprocess -d <TaskID> -c 3d_fullres -np <num.processes> -npfp <num.processes>
# Example
nnUNetv2_plan_and_preprocess -d 555 -c 3d_fullres -np 4 -npfp 4
```
The TaskID parameter has to be the one you defined in the previous step. The -np and -npfp parameters define how many processes are used during the preprocessing.  A higher number means the preprocessing is faster but more RAM is consumed and with lower numbers less RAM is needed but the processing will take longer. You can play around with this parameter, for us 4 worked well. Now that your data abides to the nnUNet format and are preprocessed, the training process can start.

# 2.4. Preprocessing of the images to be predicted 
This step aims to preprocess the images that you want to run the predictions on (i.e., the ones that were not selected as part of the training dataset). This preprocessing mainly entails an image normalization and a conversion from a 3D .tif stack to a format nnUNet can read, i.e., .nii.gz. To convert the images with the following command: 
````shell
python preprocessing_nnUNet_predict.py -i /path/to/images_mha -o /path/to/images_nii
````


2. **(Optional) Split the images:** To increase the inference speed divide the images into smaller parts to fit onto the gpu.
````shell
python preprocessing_nnUNet_predict_split.py -i <input.paht.to.imgs> -o <output.path.for.splits> -m <path.to.model> -s <num_plits>
# Also works without giving the model path, but default values are taken then
python preprocessing_nnUNet_predict_split.py -i <input.paht.to.imgs> -o <output.path.for.splits> -s <num_plits>
# Example
python preprocessing_nnUNet_predict_split.py -i /media/l727r/data/UFZ_CTPoreRootSegmentation/HI_dataset2_grass_vs_crop/test_splitting/images -o /media/l727r/data/UFZ_CTPoreRootSegmentation/HI_dataset2_grass_vs_crop/test_splitting/images_split -s 8 -m /home/l727r/Documents/cluster-checkpoints/nnUNetv2_trained_models/Dataset167_UFZ_Dataset2_grass_vs_crop_v2/nnUNetTrainer_betterIgnoreSampling_noSmooth__nnUNetPlans__3d_fullres
````
# 3. Training
# 3.1. A few words on computation on HPC Clusters
An HPC cluster is a system made up of multiple interconnected computers (often called nodes) that work together to perform complex computations. These clusters are designed to handle tasks that require a large amount of computational power, such as scientific simulations, data analysis, machine learning, or rendering. Most HPC clusters run on Linux due to its stability, scalability, and open-source nature. Linux offers better performance for parallel computing and has robust support for managing resources and running distributed applications. HPC clusters typically use a job scheduler (like Slurm) to allocate resources and manage the execution of computational tasks across the nodes. For the rest of this example, we will assume that you also dispose of a cluster running a SLURM job scheduler. 

# 3.2. Connecting to your HPC Cluster
How you connect to your HPC Cluster is mostly gonna dependent on the infrastructure and software available in your university or research institution. At the UFZ, we have prefered to use softwares with GUIs. For this, we have used FileZilla to transfer local files (i.e., the files on your workstation) to remote locations (i.e., folders on the HPC). We have also used X2GO client to connect to the head node of the cluster and send jobs across the cluster. We recommend that you contact your IT administrator to figure out what is the easiest solution for you. In the following, we will assume that got these steps right and that you could successfully connect to your HPC Cluster.

# 3.3. Setting up your HPC Cluster
## 3.3.1. Setting  up a virtual environment
Because we are now physically using another computer, we have to create (again) a new virtual environment for nnUNet. Since the cluster runs on Linux OS, the commmands to create virtual environment slightly differ. Here is an example:

````shell
python -m venv /home/username/venv-nnunet
````
where "username" is your login name and "venv-nnunet" is the name of your virtual environment. Here we used venv-nnunet to be consistent with the previous naming.

Then activate the virtual environment with the following command: 
````shell
source /home/username/venv/bin/activate
````

## 3.3.2. Install PyTorch
Once your virtual environment is created, you can now install PyTorch. PyTorch is an open-source machine learning library for Python, widely used for deep learning and artificial intelligence. It provides flexible tools for building, training, and deploying neural networks, with strong support for GPUs and dynamic computation graphs. To install PyTorch, enter the following command: 
````
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
````
Note that, here, we install an older PyTorch version (compatible with the CUDA platform 11.7). More recent versions of CUDA are currently available, however we have not yet tried them.

## 3.3.3. Install nnUNet
To install nnUNet, please repeat the operation described at the section 2.1.3. 

# 3.4. Creation of classifiers
This step trains nnUNet with the data from nnUNet_preprocessed and saves the models, logs and checkpoints in nnUNet_results.
The content of nnUNet_preprocessed is used during training. 
If preprocessing and training are done on different devices you have to sync the nnUNet_preprocessed folder to the device on which you want to train. nnUNet is trained using 5-fold cross-validation. 
This means you have to run a separate training for each fold and each fold creates a classifier file (the checkpoints_best.pth file which contains the weights of the model). The TaskID parameter is again the one you defined in the first step.




# 4. Inference
When the five training folds are completed we can use the model to make predictions.
For the predictions, the content of the nnUNet_results folder is needed. 
If training and predicting are done on different devices you have to sync the nnUNet_results folder to the device on which you want to predict.

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

To run the training on the EVE cluster, create a shell file (.sh) named (for instance) submit_nnunet_tr_fold0.sh using gedit. 

````shell
gedit submit_nnunet_tr_fold0.sh 
````
Copy then the following lines of codes within the shell file

````shell
#!/bin/bash

#SBATCH --job-name=nnunet_training  # jobname
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
source /home/username/venv-nnunet/bin/activate 
## declaring the environment variable
export nnUNet_preprocessed="/data/bosys-nnunet/nnUNet_preprocessed" 
export nnUNet_results="/work/phalempi/nnUNet_results" 

## Set the number of processes used for data augmentation PER GPU
export nnUNet_n_proc_DA=28 # the faster the GPU, the higher the value

## run nnUNet training command (see section 2.3)
nnUNetv2_train 444 3d_fullres $SLURM_ARRAY_TASK_ID -tr nnUNetTrainer_betterIgnoreSampling
````
Then submit the shell script as a sequential job using the following command. 

````shell
sbatch -a 0-4 submit_nnunet_training.sh 
````
Depending on the resources currently available, the SLURM system will distribute each training fold to a GPU, so that all the five training will run simultaneously. 

Please consider the following when requesting resources on your cluster. One of the most important resource at EVE is the maximum runtime of jobs. It specifies a limit which a running job may not exceed. If the job exceeds the requested time, it will be killed automatically by the scheduler. The same applies for the requested memory per cpu. It is a good practice to optimize these parameters to avoid exceeding the job requirements, but to keep them as low as possible so that the scheduler grants resources quicker. Note also that GPU Nodes only have a maximum of 470GB RAM available. This means that the total amount of RAM (calculated as cpus-per-task * mem-per-cpu) has to be inferior to 470GB. 

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
- **Runtime reduction:** There is some tradeoff between runtime and performance, the current setup aims for getting the best result. 
If you want to reduce the runtime, there are different ways to archive this, but with the downside of loosing some performance.
Again, just let us know if you want same changes here.

- **Format of input images:** There is the possibility to start with grayscale images in .mha format. 
- **Checkpoints during training:** Note that during training, checkpoints are automatically created after 50 epochs. If for whatever reasons a training fold killed by your scheduler, you can resume training from a previously created checkpoint. Therefore, the following command can be used.

````shell
nnUNetv2_train <TaskID> 3d_fullres <fold> -tr nnUNetTrainer_betterIgnoreSampling --c
````
# Acknowledgements
Part of this work was funded by Helmholtz Imaging (HI), a platform of the Helmholtz Incubator. 

<p align="left">
  <img src="Figures/UFZ_Logo.png" width="500"> 
  <img src="Figures/HI_Logo.png" width="300"> 
</p>
