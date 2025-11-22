# [YOLOv11-RGBT: Towards a Comprehensive Single-Stage Multispectral Object Detection Framework](https://arxiv.org/abs/2506.14696)
[![arXiv](https://img.shields.io/badge/arXiv-2506.14696-b31b1b.svg)](https://arxiv.org/abs/2506.14696)
[![One Drive Models](https://img.shields.io/badge/%F0%9F%A4%97%20One%20Drive-Models%20&%20Datasets-blue)](https://1drv.ms/f/c/384d71bb2abb0199/Eh78MfQQYMNGi1owiw4yqywBHMVzltmccCuPRfkOriALgg?e=wDKPPx)
[![Google Drive Models](https://img.shields.io/badge/%F0%9F%A4%97%20Goole%20Drive-Models%20&%20Datasets-blue)](https://drive.google.com/drive/folders/14T2OaLAiMxlx8WJVyJ2x5DLI8RNI0R8m?usp=drive_link) 
[![Baidu Drive Models](https://img.shields.io/badge/%F0%9F%A4%97%20Baidu%20Drive-Models-green)](https://pan.baidu.com/s/1Q6H98fiW_f7Kdq6-Ms6oUg?pwd=669j) 
[![Baidu Drive Datasets](https://img.shields.io/static/v1?label=Baidu%20Drive&message=Datasets&color=green)](https://pan.baidu.com/s/1xOUP6UTQMXwgErMASPLj2A?pwd=9rrf)

![FLIR_Parameters:](PaperImages/FLIR_Parameters.png)
## Introduction
This project aims to demonstrate how to configure visible and infrared datasets to accommodate multimodal object detection tasks based on YOLOv11. With three different configuration methods (directory configuration and TXT file configuration), you can easily associate visible light datasets with infrared datasets.

- YAML files for all YOLO series from YOLOv3 to YOLOv12, along with corresponding RGBT YAML files, have been added.
- The training mode of YOLOv11 itself is retained. It is recommended to learn how to set up the YOLOv11 environment and how to use it before using this project (YOLOv11 environment can be used seamlessly).

- Added YAML files for all YOLO series from YOLOv3 to YOLOv12, as well as the corresponding RGBT YAML files.
- Retained the original training mode of YOLOv11. It is recommended to first learn how to set up the environment and usage of YOLOv11 before using this project (which can seamlessly utilize the environment of YOLOv11).
- Supports multi-spectral object detection, multi-spectral keypoint detection, and multi-spectral instance segmentation tasks.
- Compared to YOLOv11, two additional parameters have been added: channels, use_simotm, and the ch in the YAML model file must correspond accordingly.
- channels: 1 # (int) Number of model channels, detailed introduction is provided below.
- use_simotm: SimOTMBBS # (str) The training mode used, such as BGR, RGBT, Gray, etc.
![YOLOv11-RGBT-RGBT:](PaperImages/YOLOv11-RGBT.jpg)

## News:
- 2025-11-22 New addition: 1. TIFF image support (for all modes)  2. 16-bit multi-spectral target detection dataset with any number of channels
- 2025-10-14 Model weights & Dataset (OneDrive): [one drive](https://1drv.ms/f/c/384d71bb2abb0199/Eh78MfQQYMNGi1owiw4yqywBHMVzltmccCuPRfkOriALgg?e=wDKPPx)
- 2025-09-17 Added the "pairs_rgb_ir" parameter. By adding the "pairs_rgb_ir" parameter, you can customize the names for visible light and infrared. The principle is to replace character 1 with character 2. By default, pairs_rgb_ir = ['visible', 'infrared']
- 2025-07-10 New additions: Download link for model weight files and datasets: [google drive](https://drive.google.com/drive/folders/14T2OaLAiMxlx8WJVyJ2x5DLI8RNI0R8m?usp=drive_link)
- 2025-07-04 New additions: Download link for model weight files [baidu drive](https://pan.baidu.com/s/1Q6H98fiW_f7Kdq6-Ms6oUg?pwd=669j) code: 669j
- 2025-06-24 New additions: YOLOv13 and YOLOv13-RGBT [paper](https://arxiv.org/abs/2506.17733) [code](https://github.com/iMoonLab/yolov13)
- 2025-06-22 Added the NiNfusion and TransformerFusionBlock modules of ICAFusion (https://github.com/chanchanchan97/ICAFusion)
- 2025-06-19 Added the MCF training code and a simple tutorial corresponding to the paper
- 2025-06-18 Correction: This framework is applicable to all pixel-aligned images, not limited to multispectral images only, but also including depth maps and SAR images, etc.
- 2025-06-18 Added the access link for the paper [YOLOv11-RGBT https://arxiv.org/abs/2506.14696](https://arxiv.org/abs/2506.14696)
- 2025-05-31 New multi-spectral object detection dataset with arbitrary number of channels
- 2025-04-18 Add CTF [CTF](https://github.com/DocF/multispectral-object-detection)
- 2025-02-14 The first submission of fully trainable and analyzable code was made.


## Supported image formats（use_simotm）:
1. uint8: 'Gray'  Single-channel 8-bit gray-scale image.  channels=1 ,  yaml   ch: 1 
2. uint16: 'Gray16bit' Single-channel 16-bit gray-scale image.  channels=1 ,  yaml   ch: 1 
3. uint8: 'SimOTM' 'SimOTMBBS'   Single-channel 8-bit gray-scale image TO Three-channel 8-bit gray-scale image.  channels=3 ,  yaml   ch: 3 
4. uint8: 'BGR'  Three-channel 8-bit color image.  channels=3 ,  yaml   ch: 3 
5. unit8: 'RGBT' Four-channel 8-bit color image.(Including early fusion, middle fusion, late fusion, score fusion, weight sharing mode)  channels=4 ,  yaml   ch: 4 
6. unit8: 'RGBRGB6C' Six-channel 8-bit color image.(Including early fusion, middle fusion, late fusion, score fusion, weight sharing mode) channels=6 ,  yaml   ch: 6 
7. unit8: 'Multispectral'  8-bit multi-spectral images for any channel (including pre-fusion, mid-fusion, post-fusion, fractional fusion, and weight-sharing mode) channels=n
8. unit16: 'Multispectral_16bit'  16-bit multi-spectral images for any channel (including pre-fusion, mid-fusion, post-fusion, fractional fusion, and weight-sharing mode) channels=n

Among them, the directory format of 1-4 is consistent with YOLOv8. With train.txt and val.txt, all you need to do is write the image address below visible, and the data format directory of 'RGBT' is as follows:


## Dataset Configuration

### 1. Dataset Structure
In YOLOv8, the visible light (visible) directory must conform to the dataset configuration principles. Additionally, an infrared (infrared) directory must exist at the same level as the visible light directory. Furthermore, the dataset should be divided into `train` and `val` (optional) subdirectories for training and validation purposes, respectively.

### 2. Configuration Methods
Below are four recommended configuration methods (Method 1 is preferred and matches the *NewVersion* dataset in the shared drive; the other three are legacy layouts for older releases):

#### Important Notes
- Ensure that the visible and infrared directories are at the same level.
- If constructing a YAML file using TXT files, the TXT file paths must include `visible` so that the program can automatically replace it with `infrared`.
- If you encounter issues, please refer to the `load_image` function in `ultralytics/data/base.py`.

---


#### Method 1: Directory Layout (FLIR example, matching the “new-version” folder in the shared drive)
Store RGB and thermal data in two peer-level folders. Each modality contains train and test sub-folders. The structure is:
```
dataset/                  # root of the real dataset; keep any name, any location
│   ├── visible/          # RGB images + labels
│   │   ├── train/        # training split
│   │   │    ├── image1.jpg 
│   │   │    ├── image1.txt   
│   │   │    ├── image2.jpg   
│   │   │    ├── image2.txt 
│   │   │    └── ...
│   │   └── test/         # val / test split
│   │        ├── image5.jpg 
│   │        ├── image5.txt   
│   │        ├── image6.jpg   
│   │        ├── image6.txt 
│   │        └── ...
│   └── infrared/         # thermal images + labels
│       ├── train/        # training split
│       │    ├── image1.jpg 
│   │   │    ├── image1.txt   
│   │   │    ├── image2.jpg   
│   │   │    ├── image2.txt 
│   │   │    └── ...
│       └── test/         # val / test split
│            ├── image5.jpg 
│            ├── image5.txt   
│            ├── image6.jpg   
│            ├── image6.txt 
│            └── ...
labels/                   # optional; can be omitted
    ├── train/
    │   ├── image1.txt
    │   └── image2.txt
    └── test/
        ├── image5.txt
        └── image6.txt

```
```
# FLIR_aligned-rgbt.yaml  RGB + thermal dataset descriptor

path: E:/BaiduNetdiskDownload/RGBTO/FLIR3C  # change to your path
train: visible/train
val:   visible/test

## absolute paths also work
# train: E:/BaiduNetdiskDownload/RGBTO/FLIR3C/visible/train
# val:   E:/BaiduNetdiskDownload/RGBTO/FLIR3C/visible/test

# number of classes
nc: 3

# class names
names: ["person", "car", "bicycle"]
```

```
# FLIR_aligned-rgb.yaml  RGB-only descriptor (same content, for clarity)

path: E:/BaiduNetdiskDownload/RGBTO/FLIR3C
train: visible/train
val:   visible/test
nc: 3
names: ["person", "car", "bicycle"]
```

```
# FLIR_aligned-inf.yaml  thermal-only descriptor

path: E:/BaiduNetdiskDownload/RGBTO/FLIR3C
train: infrared/train
val:   infrared/test
nc: 3
names: ["person", "car", "bicycle"]
```

#### Method 2: Directory Configuration (KAIST Configuration Example)
Store visible and infrared data in directories at the same level, with each modality divided into `train` and `val` subdirectories. The directory structure is as follows:

```
dataset/  # Root directory of the dataset
├── train/  # Store training data
│   ├── visible/  # Data related to visible light images
│   │   ├── images/  # Visible light image files
│   │   └── labels/  # Label files for visible light images (e.g., annotation information)
│   └── infrared/  # Data related to infrared images
│       ├── images/  # Infrared image files
│       └── labels/  # Label files for infrared images (e.g., annotation information)
└── val/  # Store validation data
    ├── visible/  # Data related to visible light images
    │   ├── images/  # Visible light image files
    │   └── labels/  # Label files for visible light images (e.g., annotation information)
    └── infrared/  # Data related to infrared images
        ├── images/  # Infrared image files
        └── labels/  # Label files for infrared images (e.g., annotation information)

---------------------------------------------------------------------

# KAIST.yaml

# train and val data as 1) directory: path/images/
train: dataset/train/visible/images  # 7601 images
val:  dataset/val/visible/images # 2257 images

# number of classes
nc: 1

# class names
names: [ 'person', ]

-----------------------------------------------------------------------
```

- **train/visible**: Stores visible light images and their labels for the training set.
- **train/infrared**: Stores infrared images and their labels for the training set.
- **val/visible**: Stores visible light images and their labels for the validation set.
- **val/infrared**: Stores infrared images and their labels for the validation set.

The program will automatically recognize visible and infrared data through the directory structure.

#### Method 3: Directory Configuration (Configuration Example)
Under the second-level directory, store visible and infrared data in directories at the same level, with each modality divided into `train` and `val` subdirectories. The directory structure is as follows:

```
dataset/
├── images/
│   ├── visible/
│   │   ├── train/  # Store training visible light images
│   │   └── val/    # Store validation visible light images
│   └── infrared/
│       ├── train/  # Store training infrared images
│       └── val/    # Store validation infrared images
└── labels/
    ├── visible/
    │   ├── train/  # Store training visible light image labels
    │   └── val/    # Store validation visible light image labels
    └── infrared/
        ├── train/  # Store training infrared image labels
        └── val/    # Store validation infrared image labels

---------------------------------------------------------------------

# KAIST.yaml

# train and val data as 1) directory: path/images/
train: dataset/images/visible/train  # 7601 images
val:   dataset/images/visible/val # 2257 images

# number of classes
nc: 1

# class names
names: [ 'person', ]

-----------------------------------------------------------------------
```

- **`images/`**: Stores all image data.
  - **`visible/`**: Contains visible light images.
    - **`train/`**: Visible light images for model training.
    - **`val/`**: Visible light images for model validation.
  - **`infrared/`**: Contains infrared images.
    - **`train/`**: Infrared images for model training.
    - **`val/`**: Infrared images for model validation.

- **`labels/`**: Stores all image label information (e.g., annotation files, comments).
  - **`visible/`**: Contains labels for visible light images.
    - **`train/`**: Labels for the training set of visible light images.
    - **`val/`**: Labels for the validation set of visible light images.
  - **`infrared/`**: Contains labels for infrared images.
    - **`train/`**: Labels for the training set of infrared images.
    - **`val/`**: Labels for the validation set of infrared images.

The program will automatically recognize visible and infrared data through the directory structure.

#### Method 4: TXT File Configuration (VEDAI Configuration Example)
Use TXT files to specify data paths. The TXT file content should include visible light image paths, and the program will automatically replace them with the corresponding infrared paths. TXT files need to specify the paths for the training and validation sets (default configuration method for YOLOv5, YOLOv8, YOLOv11).

```
dataset/
├── images/
│   ├── visible/    # Store  visible light images
│   │   ├── image1.jpg  
│   │   └── image2.jpg
│   │   └── ...      
│   └── infrared/  #  Store  visible light images
│       ├── image1.jpg   
│       └── image2.jpg  
│       └── ...         
└── labels/
    ├── visible/  # Store  visible light labels
    │   ├── image1.txt   
    │   └── image2.txt 
    └── infrared/  # Store  infrared light labels
        ├── image1.txt
        └── image2.txt    
        
---------------------------------------------------------------------

# VEDAI.yaml

train:  G:/wan/data/RGBT/VEDAI/VEDAI_train.txt  # 16551 images
val:  G:/wan/data/RGBT/VEDAI/VEDAI_trainval.txt # 4952 images

# number of classes
nc: 9

# class names
names: ['plane', 'boat', 'camping_car', 'car', 'pick-up', 'tractor', 'truck', 'van', 'others']

-----------------------------------------------------------------------
        
```

**Example TXT File Content:**

**train.txt**
```
dataset/images/visible/image1.jpg
dataset/images/visible/image2.jpg
dataset/images/visible/image3.jpg
```

**val.txt**
```
dataset/images/visible/image4.jpg
dataset/images/visible/image5.jpg
dataset/images/visible/image6.jpg
```

The program will replace `visible` with `infrared` in the paths to find the corresponding infrared images.

### 3. Principle Explanation
In the `load_image` function in `ultralytics/data/base.py`, there is a line of code that replaces `visible` with `infrared` in the visible light path. Therefore, as long as there is an infrared directory at the same level as the visible light directory, the program can correctly load the corresponding infrared data.


---

## Quick Start Guide

### 1. Clone the Project
```bash
git clone https://github.com/wandahangFY/YOLOv11-RGBT.git 
cd YOLOv11-RGBT
```

### 2. Prepare the Dataset
Configure your dataset directory or TXT file according to one of the three methods mentioned above.

### 3. Install Dependencies
(It is recommended to directly use the YOLOv11 or YOLOv8 environment that has already been set up on this computer, without the need to download again.)
```bash
# Step 1.Create a virtual environment with conda
conda create -n pt121_py38 python=3.8
conda activate pt121_py38

# Step 2: Install pytorch
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch


# Step 3: Install the remaining dependencies

pip install -r requirements.txt

# If in China, more suitable:
# pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

# Step 4: Install the environment to the system 
      #(if terminal command startup is required, 
      # or for multi-GPU training)
pip install -e .

# https://pytorch.org/get-started/previous-versions/
## CUDA 10.2
#conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch
## CUDA 11.3
#conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
## CUDA 11.6
#conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
## CPU Only
#conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cpuonly -c pytorch

## CUDA 11.8
#conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia
## CUDA 12.1
#conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=12.1 -c pytorch -c nvidia
## CPU Only
#conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 cpuonly -c pytorch
```


### 4. Run the Program
```bash
python train.py --data your_dataset_config.yaml
```
#### Explanation of Training Modes

Below are the Python script files for different training modes included in the project, each targeting specific training needs and data types.

4.1. **`train.py`**
   - Basic training script.
   - Used for standard training processes, suitable for general image classification or detection tasks.

2. **`train-rtdetr.py`**
   - Training script for RTDETR (Real-Time Detection Transformer).

3. **`train_Gray.py`**
   - Grayscale image training script.
   - Specifically for processing datasets of grayscale images, suitable for tasks requiring image analysis in grayscale space.

4. **`train_RGBRGB.py`**
   - RGB-RGB image pair training script.
   - Used for training with two sets of RGB images simultaneously, such as paired training of visible and infrared images, suitable for multimodal image analysis.

5. **`train_RGBT.py`**
   - RGB-T (RGB-Thermal) image pair training script.
   - Used for paired training of RGB images and thermal (infrared) images, suitable for applications requiring the combination of visible light and thermal imaging information.

### 5. Testing
Run the test script to verify if the data loading is correct:
```bash
python val.py
```

### 6. Visualization
#### 6.1 Feature map visualization
Run the detect script for feature map visualization, set visualize=True:
```python
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r"runs/M3FD/M3FD_IF-yolo11n2/weights/best.pt") # select your model.pt path
    model.predict(source=r'G:\wan\data\RGBT\M3FD_Detection\images_coco\infrared\trainval',
                  imgsz=640,
                  project='runs/detect',
                  name='exp',
                  show=False,
                  save_frames=False,
                  use_simotm="RGB",
                  channels=3,
                  save=False,
                  # conf=0.2,
                  visualize=True # visualize model features maps
                )
```
```bash
python detect.py
```
![img_feature_map_visualization](PaperImages/img_feature_map_visualization.png)


#### 6.2 Gradcam: Heatmap visualization
Run the heatmap_RGBT.py script for heatmap visualization:
```bash
python heatmap_RGBT.py
```
![img_heatmap_visualization](PaperImages/img_heatmap_visualization.jpg)

---

## Important Notes (Emphasized Again)
- Ensure that the visible and infrared directories are at the same level, and there are `train` and `val` subdirectories under each modality.
- TXT file paths must include `visible` so that the program can automatically replace it with `infrared`.
- If you encounter issues, please refer to the `load_image` function in `ultralytics/data/base.py`.

---
# Dataset Download Links

Here are the Baidu Netdisk links for the converted VEIAI, LLVIP, KAIST, M3FD datasets (you need to change the addresses in the yaml files. If you use txt files to configure yaml files, you need to replace the addresses in the txt files with your own addresses: open with Notepad, Ctrl+H). (Additionally, if you use the above datasets, please correctly cite the original papers. If there is any infringement, please contact the original authors, and it will be removed immediately.)

- VEIAI (Vehicle Detection in Aerial Imagery (VEDAI) : a benchmark (greyc.fr))
- LLVIP (bupt-ai-cz/LLVIP: LLVIP: A Visible-infrared Paired Dataset for Low-light Vision (github.com))
- KAIST
  - Original address (SoonminHwang/rgbt-ped-detection: KAIST Multispectral Pedestrian Detection Benchmark [CVPR '15] (github.com))
  - Download of the complete and cleaned KAIST dataset - kongen - CNBlogs (cnblogs.com)
- M3FD (JinyuanLiu-CV/TarDAL: CVPR 2022 | Target-aware Dual Adversarial Learning and a Multi-scenario Multi-Modality Benchmark to Fuse Infrared and Visible for Object Detection (github.com))

Baidu Netdisk Link:
Link: https://pan.baidu.com/s/1xOUP6UTQMXwgErMASPLj2A Extraction Code: 9rrf

# Download link for model weight files
- If you use the following weights, please correctly cite the corresponding paper of this project, the YOLOv11 project or paper (as per the YOLOv11 copyright notice), and the paper corresponding to the dataset (such as the LLVIP copyright notice).

- link:https://pan.baidu.com/s/1Q6H98fiW_f7Kdq6-Ms6oUg   code:669j


## Project Structure Explanation
| Name                             | Description                                                                                                |
| -------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| **Directories**                  |                                                                                                            |
| `.github`                        | Stores GitHub-related configuration files for GitHub Actions CI/CD pipelines.                              |
| `PaperImages`                    | Stores images or charts related to the paper.                                                              |
| `docker`                         | Stores Docker-related configurations and scripts for project containerization.                             |
| `docs`                           | Stores project documentation, such as user guides and API docs.                                            |
| `examples`                       | Provides example code or configurations to help users get started quickly.                                 |
| `tests`                          | Contains test code for verifying project functionality.                                                    |
| `ultralytics`                    | Contains code or configurations related to Ultralytics, the development team behind the YOLO model series. |
| **Files**                        |                                                                                                            |
| `.dockerignore`                  | Specifies files or directories to ignore when building a Docker image.                                     |
| `.gitignore`                     | Defines patterns for files or directories to ignore in Git version control.                                |
| `CITATION.cff`                   | Provides citation information for the project.                                                             |
| `CONTRIBUTING.md`                | Offers guidelines for contributing to the project.                                                         |
| `LICENSE`                        | Contains license information for the project, outlining legal terms for use and distribution.              |
| `README.md`                      | Project documentation file, including project introduction and usage instructions.                         |
| `README_Zh.md`                   | Chinese version of the README for Chinese-speaking users.                                                  |
| `YOLOv11-RGBT-2506.14696_v2.pdf` | Project-related PDF document, including papers and detailed explanations.                                  |
| `detect-1C.py`                   | Example script for single-channel detection tasks.                                                         |
| `detect-4C.py`                   | Script for four-channel detection tasks.                                                                   |
| `detect-6C.py`                   | Script for six-channel detection tasks.                                                                    |
| `detect-multispectral.py`        | Script for multispectral detection.                                                                        |
| `detect.py`                      | General-purpose detection script.                                                                          |
| `export.py`                      | Script for model export functionalities.                                                                   |
| `get_FPS.py`                     | Measures the model's frame rate (FPS).                                                                     |
| `heatmap_RGBT.py`                | Script for generating heatmaps, potentially for visualizing detection results.                             |
| `mkdocs.yml`                     | Configuration file for MkDocs to generate project documentation.                                           |
| `pyproject.toml`                 | Build configuration file for Python projects.                                                              |
| `requirements.txt`               | Lists Python packages and versions required by the project.                                                |
| `train-rt detr.py`               | Training script for the RT-DETR model.                                                                     |
| `train.py`                       | General-purpose training script.                                                                           |
| `train_Gray.py`                  | Training script for grayscale images.                                                                      |
| `train_MCF_demo.py`              | Demo training script for the MCF strategy.                                                                 |
| `train_RGBRGB.py`                | Training script for RGB + infrared 6-channel images.                                                       |
| `train_RGBT.py`                  | Training script for RGBT (4-channel) images.                                                               |
| `train_RGBT_mine_print.py`       | Prints the parameter and computation amounts of the model in batches.                                      |
| `train_multispectral.py`         | Training script for multispectral data with arbitrary channels.                                            |
| `transform_COCO_to_RGBT.py`      | Converts pre-trained network weights from COCO to RGBT model weights.                                      |
| `transform_MCF.py`               | Script related to converting the MCF strategy.                                                             |
| `transform_PGI.py`               | Script related to converting the PGI strategy.                                                             |
| `val_PGI.py`                     | Validation script for the PGI strategy.                                                                    |
| `val.py`                         | Model validation script.                                                                                   |


## Contributions
PRs or Issues are welcome to jointly improve the project. This project is a long-term open-source project and will continue to be updated for free in the future, so there is no need to worry about cost issues.

## Contact Information
- GitHub: [https://github.com/wandahangFY](https://github.com/wandahangFY)
- Email: wandahang@foxmail.com
- QQ: 1753205688
- QQ Group: 483264141 (Free)

![QQ Group](PaperImages/QQ.png)


## Chinese Interpretation Link
- [Modified YOLOv8 for RGBT multi-channel and single-channel gray image detection  ](https://zhuanlan.zhihu.com/p/716419187)

## Video Tutorial Link
- [Video Tutorial and Secondary Innovation Solutions for YOLO-MIF]() [TODO: Detailed tutorial in text-based PPT format]

## Secondary Innovation Points Summary and Code Implementation (TODO)
- [Secondary Innovation Solutions]() [The last page of the PPT tutorial provides some secondary innovation solutions. TODO: Will be written and updated later if needed]

## Paper Link
[YOLOv11-RGBT https://arxiv.org/abs/2506.14696](https://arxiv.org/abs/2506.14696)

[YOLO-MIF: Improved YOLOv8 with Multi-Information fusion for object detection in Gray-Scale images]( https://www.sciencedirect.com/science/article/pii/S1474034624003574)


## Citation Format
D. Wan, R. Lu, Y. Fang, X. Lang, S. Shu, J. Chen, S. Shen, T. Xu, Z. Ye, YOLOv11-RGBT: Towards a Comprehensive Single-Stage Multispectral Object Detection Framework, (2025). https://doi.org/10.48550/arXiv.2506.14696.

@misc{wan2025yolov11rgbtcomprehensivesinglestagemultispectral,
      title={YOLOv11-RGBT: Towards a Comprehensive Single-Stage Multispectral Object Detection Framework}, 
      author={Dahang Wan and Rongsheng Lu and Yang Fang and Xianli Lang and Shuangbao Shu and Jingjing Chen and Siyuan Shen and Ting Xu and Zecong Ye},
      year={2025},
      eprint={2506.14696},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2506.14696}, 
}

## Reference Links
- [Codebase used for overall framework: YOLOv8](https://github.com/ultralytics/ultralytics)
- [Reparameterization reference code by Ding Xiaohan: DiverseBranchBlock](https://github.com/DingXiaoH/DiverseBranchBlock)
- [Some modules reference from Devil Mask's open-source repository](https://github.com/z1069614715/objectdetection_script)
- [YOLOv7](https://github.com/WongKinYiu/yolov7)
- [Albumentations Data Augmentation Library](https://github.com/albumentations-team/albumentations)
- [YOLO-AIR](https://github.com/iscyy/yoloair)
- [CTF](https://github.com/DocF/multispectral-object-detection)

## Closing Remarks
Thank you for your interest and support in this project. The authors strive to provide the best quality and service, but there is still much room for improvement. If you encounter any issues or have any suggestions, please let us know.
Furthermore, this project is currently maintained by the author personally, so there may be some oversights and errors. If you find any issues, feel free to provide feedback and suggestions.

## Other Open-Source Projects
Other open-source projects are being organized and released gradually. Please check the author's homepage for downloads in the future.
[Homepage](https://github.com/wandahangFY)

## FAQ
1. Added README.md file (Completed)
2. Detailed tutorials (README.md)
3. Project environment setup (The entire project is based on YOLOv8 version as of November 29, 2023, configuration referenced in README-YOLOv8.md file and requirements.txt)
4. Explanation of folder correspondences (Consistent with YOLOv8, hyperparameters unchanged) (TODO: Detailed explanation)
5. Summary of secondary innovation points and code implementation (TODO)
6. Paper illustrations:
   - Principle diagrams, network structure diagrams, flowcharts: PPT (Personal choice, can also use Visio, Edraw, AI, etc.)
   - Experimental comparisons: Orgin (Matlab, Python, R, Excel all applicable)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=wandahangFY/YOLOv11-RGBT&type=Date)](https://star-history.com/#wandahangFY/YOLOv11-RGBT&Date)