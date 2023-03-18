# ADEX-Project
The scope of this research is the use of portable X-ray camera equipment suitable for explosive ordnance disposal robots. To store image data used to develop data analysis and machine learning models, including the use of X-ray image data obtained from various cases. These show the presence of explosive-unit elements according to the experience of experts from the Institute of Defense Technology and Royal Thai Air Force: Ordnance Department. To accomplish this goal, the proposed research brings about an effective application of modern artificial intelligence and machine learning technologies in pattern recognition such as deep learning.
| <img src="https://github.com/KwinLook/ADEX-Project/blob/main/Project-Diagram/General-ADEX-Diagram-Page-4.drawio.png" class="img-responsive"> |
|:---:|
| General framework of model development and evaluation |

Convolutional layered CNN model One or more layers that alternate with the integration layers are added on top of the fully connected network. It can be conceptually illustrated by the layer-wise framework in Figure 5. Typically, a distorted layer consists of many property detectors. The so-called filters or kernels that are primarily intended to capture the properties of each input image.These filters can be perceived as a matrix of weights that scroll through the image. Distortion is achieved by calculating the dot product of the matrix and the corresponding part of the image at all positions. And every result is seen as an element in a new matrix called a feature tree. This is usually followed by an aggregation layer. where the bundled features received will be divided into sections. of the same size (e.g. 2 × 2), then one input value is subsampled from all regions in terms of maximum or average age. or even random values. Therefore, the subsampling function can continuously reduce the size of the input representation to reduce network complexity. This will help speed up the calculations. and makes the network strong against minor changes, distortions, and translations as well. In our work, a maximum integration strategy is used to preserve the most important attributes of the input. There can also be a dropout layer between convolutional layers. and a dense layer or two dense layers especially It is a technique used to avoid overfit by randomly ignoring some nodes from the network in each training sample. with a predefined probability p ∈ (0, 1) e.g.  p= 0.75, 0.5, 0.25
| <img src="https://github.com/KwinLook/ADEX-Project/blob/main/Project-Diagram/ADEX-CNN%20Model.PNG" class="img-responsive"> |
|:---:|
| Processing stages in a common CNN model |

## Performance evaluation
The Confusion Matrix is a vital table to measure the ability of a machine learning model to solve any particular classification problems. It represents measures of comparison between true classes and predictions made by the model of interest. Specific to a binary classification, it contains the following four measures:
1. **TP** :True Positive mean the program indicates is an explosive and its reality is explosive
2. **TN** :True Negative mean the program indicates is non-explosive, and its reality is non-explosive 
3. **FP** :False Positive mean the program indicates is explosive, and its reality is non-explosive 
4. **FN** :False negative mean the program indicates is non-explosive, and its reality is explosive

| <img src="https://github.com/KwinLook/ADEX-Project/blob/main/Project-Diagram/Confusion%20Matrix.PNG" class="img-responsive"> |
|:---:|
| Illustration of a confusion matrix for this binary classification problem |


## To Beging With:
1. You can just click on this
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KwinLook/ADEX-Project/blob/main/Project-Code/CNN%20Model/ADEX_Project_Launch_Colab.ipynb) to open and work with the provided tutorial using a free GPU.
 
2. As for the volume of data to be increased, it must run the Python commands. [Data Augment](https://github.com/KwinLook/ADEX-Project/blob/main/Project-Code/Data%20Augmentation/ADEX_Generate_Images_Useing_DataAugment.py)
