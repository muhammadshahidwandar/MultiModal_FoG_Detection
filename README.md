# Freezing of Gait detection in Parkinson’s Disease using multi-modal data (EEG, EMG, IMU and Skin Conductance) 
# Temporal 3-Layer Resnet model using Pytorch: 

## Overview
![BlockDiagram](https://github.com/muhammadshahidwandar/MultiModal_FoG_Detection/blob/main/images/Fog_Main_Block.jpg)

![ResultsIMU](https://github.com/muhammadshahidwandar/MultiModal_FoG_Detection/blob/main/images/IMU_FoG_detection.jpg)
FoG detection using IMU multivariate signal

![ResultsEMG](https://github.com/muhammadshahidwandar/MultiModal_FoG_Detection/blob/main/images/EMG_FoG_detection.jpg)
FoG detection using EMG multivariate signal
## Sub-directories and Files
There are two sub-directories described as follows::

### images
Containes over all block diagram and visual results of predicted signal.
### source
Contains source code for data read functions, multivariate signal preprocessing utilities, and a 3-layer temporal ResNet model for classification.


## Dependencies
* python 3.7
* pandas 2.0.2
* torch 1.13.0 
* scipy  1.9.3


## Dataset
The data is downloaded from the link [https://data.mendeley.com/datasets/r8gmbtv7w2/3], which is referenced in the paper.



## Reference

**Multimodal Data for the Detection of Freezing of Gait in Parkinson’s Disease**  
```
@article{zhang2022multimodal,
  title={Multimodal Data for the Detection of Freezing of Gait in Parkinson’s Disease},
  author={Zhang, Wei and Yang, Zhuokun and Li, Hantao and Huang, Debin and Wang, Lipeng and Wei, Yanzhao and Zhang, Lei and Ma, Lin and Feng, Huanhuan and Pan, Jing and others},
  journal={Scientific Data},
  volume={9},
  number={1},
  pages={606},
  year={2022},
  publisher={Nature Publishing Group UK London}
}
