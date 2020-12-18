# Corn Image Instance Segmentation Using Mask RCNN


This repository is our implementation of our research article "Estimating Consumption of Corn Cob by Wild Animals withInstance Segmentation of Images using Mask-RCNN". Here source code and instructions for testing the models can be found. 

This repository uses implementation of [Mask RCNN](https://arxiv.org/abs/1703.06870) by [Matterport](https://github.com/matterport). For detailed setup instructions of MaskRCNN you may refer [their repository](https://github.com/matterport/Mask_RCNN).

1. Prerequisites

1.1. Deep learning framework configuration

Models used in the study were trained using Tensorflow 1.12, Please install the Matterport's MaskRCNN implementation before proceeding to test this repository.


1.2. Pretrained Mask RCNN models for corn segmentation

Download the model pretrained for segmentation from dropbox link https://figshare.com/s/94fa3750aaae4fde3ffb.

1.3. Source code for seedling detection and counting

Download the source code from this repository to a local computer where MaskRCNN implementation is installed. 


2. Instructions for Segmentation and generating consumption.

Our Testing environment: Ubuntu, Python 3.6.7, TensorFlow 1.12, NVIDIA GTX1080-Ti. 

2.1. Install necessary packages: mentioned in requirements.txt

2.2. Copy the corn folder to Matterports Home directory. (e.g., /home/sadke/MaskRcnn/corn).

2.3. corn_2class.py and corn.py files define the model configuration for training and inference for Approach 1 and Approach 2 respectively.

2.4. Run detectCorn.py to test the individual model on images in given input directory. Please see the program header for detailed explanation of changable parameters.

2.5. Run consumptio.py to test the models. Please see the program header for detailed explanation of changable parameters.

2.6. For batch computation of PR curves and obtaining PR values, Run compute_batch_IOU.py.

2.7. Run overlaps_2class.py, which can be alternatively used for batch computation of overlaps i.e. IoU for all test images.

3. Instructions for obtaining results figures (under the folder figure_programs).

Programs for generating evaluation metrics presented in paper can be tested using the results generated in section 2.

4. Additional resources

Original corn images and annotations can be downloaded via the link https://figshare.com/s/3513f21c93e102502500. All annotations were made using VGG Image Annotator 2.0.1 version.
