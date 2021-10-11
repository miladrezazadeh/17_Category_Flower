# Project Title
Convolutional Neural Network to categorize a given flower image into its respective category.

## Overview

Content:
Authors: Maria-Elena Nilsback and Andrew Zisserman
Description from the authors:

"We have created a 17 category flower dataset with 80 images for each class. The flowers chosen are some common flowers in the UK. The images have large scale, pose and light variations and there are also classes with large variations of images within the class and close similarity to other classes".

## Goals

Build a neural network, using Keras, which will categorize a given flower image into its respective category.

Try to address the problem that this data set has: it's too small. Overfitting is a major problem with the neural networks applied to this data set. To attempt to address this problem, explore various ways of addressing overfitting:

-Experiment with creating the smallest network you reasonably can.
-Explore the ability to create new, artificial data, by using the ImageDataGenerator class, which can be found in the keras.preprocessing.image subpackage. Use this enlarged data set to train your model.
-Experiment with regularization or dropout.

## Datasets
This data set consists of colour images of flowers, each of which is categorized into one of 17 categories. 
In the modified version of the data set, the images have been scaled to be 50 x 50 pixels each, rather than their original dimensions.
The original dataset : https://www.robots.ox.ac.uk/~vgg/data/flowers/17/index.html
modified dataset: https://support.scinet.utoronto.ca/education/get.php/50x50flowers.images.npy
modified target dataset: https://support.scinet.utoronto.ca/education/get.php/50x50flowers.targets.npy



