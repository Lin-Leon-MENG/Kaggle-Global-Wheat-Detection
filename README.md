# Kaggle-Global-Wheat-Detection
Kaggle Image Recognition Competition

## Team Members

* [Xiao Chu](https://github.com/vivianchu30)
* [Lin Meng](https://github.com/Lin-Leon-MENG)

# Project Objectives

The main task is how to accurately detect wheat heads in a picture, in essence, an object detection problem. By the end of this project, we would like to learn to build R-CNN and CNN models to help improve the accuracy of wheat recognition and get hands-on experiences with the end-to-end image recognition workflow. 

Our goal is to have a model that improves the current detection methods in maybe one or two aspects. Ultimately, these models can be applied to agricultural science, where simply by looking at the wheat images taken from the field, farmers can better estimate the wheat growth and quantity, therefore make better decisions over wheat production.

# What Data

The data we are planning to use are from the Kaggle Competition dataset. The datasets are separated to train and test. The train dataset contains 3422 wheat images, the test dataset contains 10 images. The total memory size of the datasets is 613.76MB. Along with the images data we also have the tabular data where all the bounding boxes describing wheat heads are recorded.

Since we have too few test examples and the overall number of images may not be enough for a deep neural networks model. We plan to split a portion of training images as validation and test set, and apply data augmentation to the new training set. We would also pay attention to the source of the images as the hidden test images are from different organizations. So reducing variance of the model will be a key factor.

# Techniques Overview

We are planning to implement R-CNN and CNN in the dataset. Because R-CNN has better performance in the region and objective detection, especially in detecting the color, texture, size, and shape of images, while CNN is a more mature supervised algorithm, which could automatically detect important features and have better performance in multi-class classification problems. 

Specifically, we will start from a baseline R-CNN model, then move into the current detection methods such as Yolo-3 and Faster-RCNN, and finally try to propose one or two improvements based on our findings to achieve better performance and generalizability.

# References

https://www.kaggle.com/c/global-wheat-detection/overview
Redmon et. al., 2015. You Only Look Once: Unified, Real-Time Object Detection
Ren et. al., 2016. Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks

# Schedule, Timeline, and Team Responsibilities

June 2nd: EDA (Lin); GPU set-up (Xiao)
June 7th: Finish data augmentation, have a baseline model (Lin, Xiao)
June 12th: Have a Yolo (Lin) or Faster RCNN (Xiao) model, progress report (Lin, Xiao)
June 19th: Fine-tuning, focus on one direction for improvement (Lin, Xiao)
June 22nd: Final report (Lin, Xiao)
June 24th: Final presentation

