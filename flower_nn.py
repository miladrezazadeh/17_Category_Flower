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
import warnings
warnings.filterwarnings("ignore")

#################################################################
## Read the flower dataset, images and targets
print("Reading flowers input file.")
images = np.load('50x50flowers.images.npy')
print("Reading flowers target file.")
targets = np.load('50x50flowers.targets.npy')
print(targets.shape)

