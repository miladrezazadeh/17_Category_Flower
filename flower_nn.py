## Name: Milad Rezazadeh
## SciNet username: rezaza10
## Description:
##  Neural Network which will categorize a given flower image
## into its respective category

#################################################################

## Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras.models as km, keras.layers as kl
import os
import urllib
import sklearn.model_selection as skms
import warnings
warnings.filterwarnings("ignore")

#################################################################

## Read the flower dataset, images and targets
print("Reading flowers input file.")
images = np.load('50x50flowers.images.npy')
print("Reading flowers target file.")
targets = np.load('50x50flowers.targets.npy')
## targets are converted to np array (1360,1)
targets = np.asarray(targets).reshape(1360,1)

#################################################################

## split data into training and test
from sklearn.model_selection import train_test_split
train_images, test_images, train_targets, test_targets = train_test_split(images,
targets, train_size = 0.8, random_state = 42)

################################################################3


