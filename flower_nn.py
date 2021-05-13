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

#################################################################

## Prepping the data
## The targets also need to be changed to categorical format
import tensorflow.keras.utils as ku
train_targets = ku.to_categorical(train_targets, 17)
test_targets = ku.to_categorical(test_targets, 17)

#################################################################

## Building the NN model
## Modified from lecture5_code
def get_model(numfm, numnodes, input_shape = (50, 50, 3),
              output_size = 17):

    """
    This function returns a convolution neural network Keras model,
    with numfm feature maps and numnodes neurons in the
    fully-connected layer.

    Inputs:
    - numfm: int, the number of feature maps in the convolution layer.

    - numnodes: int, the number of nodes in the fully-connected layer.

    - intput_shape: tuple, the shape of the input data,
    default = (50, 50, 3).

    - output_size: int, the number of nodes in the output layer,
      default = 17.

    Output: the constructed Keras model.

    """

    ## Initialize the model.
    model = km.Sequential()

    ## Add a 2D convolution layer, with numfm feature maps.
    model.add(kl.Conv2D(numfm, kernel_size = (5, 5),
                        input_shape = input_shape,
                        activation = 'relu'))

    ## Add a max pooling layer.
    model.add(kl.MaxPooling2D(pool_size = (2, 2),
                              strides = (2, 2)))

    ## Convert the network from 2D to 1D.
    model.add(kl.Flatten())

    ## Add a fully-connected layer.
    model.add(kl.Dense(numnodes,
                       activation = 'tanh'))

    ## Add the output layer.
    model.add(kl.Dense(10, activation = 'softmax'))

    ## Return the model.
    return model

#################################################################

