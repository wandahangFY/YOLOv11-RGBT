# YOLO-MIF: Improved YOLOv8 with Multi-Information Fusion for Object Detection in Gray-Scale Images

## Introduction
This paper proposes an enhanced object detection network, YOLO-MIF, for addressing the challenges of object detection in gray-scale images. The network integrates multiple multi-information fusion strategies to improve the YOLOv8 network. The paper first introduces a technique for creating pseudo multi-channel gray-scale images to increase the network's channel information and alleviate potential image noise and defocus blur issues. Subsequently, by using network structure reparameterization techniques, the detection performance of the network is improved without increasing the inference time. Additionally, a novel decoupled detection head is introduced to enhance the model's expressive power when dealing with gray-scale images. The algorithm is evaluated on two open-source gray-scale image detection datasets (NEU-DET and FLIR-ADAS). The results show that at the same speed, the algorithm outperforms YOLOv8 by 2.1% and Faster R-CNN by 4.8% in balancing detection efficiency and effectiveness.

![YOLO-MIF principle diagram as follows:](PaperImages/YOLO-MIF.png)

## Contributions
1. YOLO-MIF: An object detection network designed for gray-scale images
2. New reparameterization modules: WDBB, RepC2f
3. Rep3C Head
4. GIS: Input strategy for gray-scale images

## Supported image formats:
1. uint8: 'Gray'  Single-channel 8-bit gray-scale image.
2. uint16: 'Gray16bit' Single-channel 16-bit gray-scale image.
3. uint8: 'SimOTM' 'SimOTMBBS'   Single-channel 8-bit gray-scale image TO Three-channel 8-bit gray-scale image.
4. uint8: 'BGR'  Three-channel 8-bit color image.
5. unit8: 'RGBT' Four-channel 8-bit color image.(Including early fusion, middle fusion, late fusion, score fusion, weight sharing mode)

Among them, the directory format of 1-4 is consistent with YOLOv8. With train.txt and val.txt, all you need to do is write the image address below visible, and the data format directory of 'RGBT' is as follows:
![img.png](img.png)

![YOLO-MIF-RGBT:](PaperImages/YOLO-MIF-RGBT.jpg)

## Installation
<details open>
<summary>Install</summary>

Pip install the ultralytics package including all [requirements](https://github.com/ultralytics/ultralytics/blob/main/requirements.txt) in a [**Python>=3.7**](https://www.python.org/) environment with [**PyTorch>=1.7**](https://pytorch.org/get-started/locally/).

```bash
pip install -r requirements.txt
```

</details>


<details open>
<summary>Usage</summary>

1. NEU-DET 
```bash
python train_NEU-DET-RepDC.py 

```

2. FLIR-ADAS
```bash
python train_FLIR_ADAS-16-RepDCHead.py

```

</details>

<details open>
<summary>Correspondence between Paper and Code</summary>

1. RIR=True + SimOTMBBS = GIS
- SimOTM yields better results but reduces speed, while the SimOTMBBS used in this paper almost does not reduce speed. If readers need, SimOTM will be open-sourced separately on arXiv without further journal submissions. Original paper and details can be found at: [Link](https://www.researchgate.net/publication/372944004_Otm-Fusion_An_Image_Preprocessing_Method_for_Object_Detection_in_Grayscale_Image)
- Function.cpp contains CUDA and C++ (CPU) implementations
- Code related to GIIS can be found in ultralytics/yolo/data/base.py
- Code related to NEU-DET can be found in train_NEU-DET-RepDC.py
- train-gray.py for single channel training and inference --use_simotm is 'Gray' or 'Gray16bit', channels=1, Model files inside need to set up ch: 1 see ultralytics/models/v8 / yolov8-Gray.yaml
- train_RGBT.py for multi-channel training and inference --use_simotm is 'RGBT', channels=4, In the model file you need to set ch:4 see ultralytics/models/v8-RGBT/yolov8-RGBT-earlyfusion.yaml

```python
parser.add_argument('--use_simotm', type=str, choices=['Gray2BGR', 'SimOTM', 'SimOTMBBS','Gray','SimOTMSSS','Gray16bit','BGR','RGBT'], default='SimOTMBBS', help='simotm')
parser.add_argument('--channels', type=int, default=3, help='input channels')
```
- GIS 
![GIS simplified diagram as follows:](PaperImages/GIS.png)

2. Reparameterization Modules

- Code related to WDBB can be found in ultralytics/nn/modules/rep_block.py
```python
['DiverseBranchBlock','DeepACBlockDBB','WideDiverseBranchBlock','DeepDiverseBranchBlock','ACBlockDBB','ACBlock']
# WideDiverseBranchBlock corresponds to WDBB mentioned in the paper, other modules need further experimentation and verification
```
- WDBB 
![WDBB simplified diagram as follows:](PaperImages/WDBB.png)

- DeepDBB (experimental and theoretical details not explained in the paper)
![DeepDBB simplified diagram as follows:](PaperImages/DeepDBB.png)


- Code related to RepC2f can be found in ultralytics/nn/modules/block.py
```python
'C2f_ACDBB', 'C2f_DeepACDBB', 'C2f_DeepDBB', 'C2f_DeepACDBBMix', 'C2f_DBB', 'C2f_ACNET', 'C2f_WDBB'

# C2f_WDBB in the code corresponds to RepC2f in the paper, details about C2f_DeepDBB will be used in the next paper. Others need further experimentation and verification
```

- Code related to Rep3C Head can be found in ultralytics/nn/modules/head.py
```python
'Detect', 'Segment', 'Pose', 'Classify', 'RTDETRDecoder','DetectDBB','DetectACDBB','DetectAC','DetectDeepDBB',\
          'DetectDeepACDBB' , 'Detect_Efficient','DetectSingleDBB','Detect2AC2DBB',\
          'Detect2DBB2AC','Detect2DBBAC','Detect2ACDBB','Detect_Efficient3DBB','Detect_Efficient3DBBR'

# Detect_Efficient3DBB in the code corresponds to Rep3C Head in the paper, some modules have been validated effectively but not included in the paper yet. Others need further experimentation and verification
```
- Rep3C Head 
![Rep3C Head simplified diagram as follows:](PaperImages/Rep3CHead.png)

</details>
  

## Chinese Interpretation Link
- [Chinese Interpretation of YOLO-MIF](Chinese Interpretation Link) [TODO: Will be written and updated later if needed]
- [Modified YOLOv8 for RGBT multi-channel and single-channel gray image detection  ](https://zhuanlan.zhihu.com/p/716419187)

## Video Tutorial Link
- [Video Tutorial and Secondary Innovation Solutions for YOLO-MIF]() [TODO: Detailed tutorial in text-based PPT format]

## Secondary Innovation Points Summary and Code Implementation (TODO)
- [Secondary Innovation Solutions]() [The last page of the PPT tutorial provides some secondary innovation solutions. TODO: Will be written and updated later if needed]

## Paper Link
[YOLO-MIF: Improved YOLOv8 with Multi-Information fusion for object detection in Gray-Scale images]( https://www.sciencedirect.com/science/article/pii/S1474034624003574)

[https://www.sciencedirect.com/science/article/pii/S1474034624003574]( https://www.sciencedirect.com/science/article/pii/S1474034624003574)

## Citation Format
Wan, D.; Lu, R.; Hu, B.; Yin, J.; Shen, S.; xu, T.; Lang, X. YOLO-MIF: Improved YOLOv8 with Multi-Information Fusion for Object Detection in Gray-Scale Images. Advanced Engineering Informatics 2024, 62, 102709, doi:10.1016/j.aei.2024.102709.


## Reference Links
- [Codebase used for overall framework: YOLOv8](https://github.com/ultralytics/ultralytics)
- [Reparameterization reference code by Ding Xiaohan: DiverseBranchBlock](https://github.com/DingXiaoH/DiverseBranchBlock)
- [Some modules reference from Devil Mask's open-source repository](https://github.com/z1069614715/objectdetection_script)
- [YOLOv7](https://github.com/WongKinYiu/yolov7)
- [Albumentations Data Augmentation Library](https://github.com/albumentations-team/albumentations)
- Reparameterization validation code references from Handwritten AI's reparameterization course

## Closing Remarks
Thank you for your interest and support in this project. The authors strive to provide the best quality and service, but there is still much room for improvement. If you encounter any issues or have any suggestions, please let us know.
Furthermore, this project is currently maintained by the author personally, so there may be some oversights and errors. If you find any issues, feel free to provide feedback and suggestions.

## Other Open-Source Projects
Other open-source projects are being organized and released gradually. Please check the author's homepage for downloads in the future.
[Homepage](https://github.com/wandahangFY)

## FAQ
1. Added README.md file (Completed)
2. Detailed tutorials (TODO)
3. Project environment setup (The entire project is based on YOLOv8 version as of November 29, 2023, configuration referenced in README-YOLOv8.md file and requirements.txt)
4. Explanation of folder correspondences (Consistent with YOLOv8, hyperparameters unchanged) (TODO: Detailed explanation)
5. Summary of secondary innovation points and code implementation (TODO)
6. Paper illustrations:
   - Principle diagrams, network structure diagrams, flowcharts: PPT (Personal choice, can also use Visio, Edraw, AI, etc.)
   - Experimental comparisons: Orgin (Matlab, Python, R, Excel all applicable)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=wandahangFY/YOLO-MIF&type=Date)](https://star-history.com/#wandahangFY/YOLO-MIF&Date)