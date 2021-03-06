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
import pickle
import keras.models as km, keras.layers as kl
import os
import urllib
import sklearn.model_selection as skms
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings('always')

print("Using Tensorflow backend.")

#######################################################################################

## Function for downloading datasets from url
## this snippet of code is derived from original source code:
## https://github.com/tflearn/tflearn/blob/master/tflearn/datasets/cifar10.py

def download(filename, source_url, work_directory):
    if not os.path.exists(work_directory): #check if the folder exists; if not make dir
        os.mkdir(work_directory)
    filepath = os.path.join(work_directory, filename)
    if not os.path.exists(filepath): # check if file exists; if not, download
        print("Downloading file, Please wait...")
        filepath, _ = urllib.request.urlretrieve(source_url + filename, # this is a function to download files
                                                 filepath)
        statinfo = os.stat(filepath)
        print(('Succesfully downloaded', filename, statinfo.st_size, 'bytes.'))
    return filepath


## Download the input dataset
#download("50x50flowers.images.npy",
        #  "https://support.scinet.utoronto.ca/education/get.php/50x50flowers.images.npy",
        # "17_Category_Flower_input")

## Download the target dataset
#download("50x50flowers.targets.npy",
      # "https://support.scinet.utoronto.ca/education/get.php/50x50flowers.targets.npy",
      #  "17_Category_Flower_input")

#######################################################################################

## Read the flower dataset, images and targets

print("Reading flowers input file.")
images = np.load('17_Category_Flower_input/50x50flowers.images.npy',
                 allow_pickle=True, fix_imports=True, encoding='latin1')

print("Reading flowers target file.")
targets = np.load('17_Category_Flower_input/50x50flowers.targets.npy',
                  allow_pickle=True, fix_imports=True, encoding='latin1')

#######################################################################################

## split data into training and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(images, targets,
                                                    test_size = 0.25, random_state = 42)

######################################################################################
## Normalizing the training data
x_train = x_train / 255.0
x_test = x_test / 255.0

######################################################################################

## Prevent overfitting with Data augmentation
## this snippet of code is derived from original source code:
## https://www.kaggle.com/rajmehra03/flower-recognition-cnn-keras
## we are trying to generate more (fake) data with random rotation angle, zoom, flip, etc.

from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(x_train)

######################################################################################

## Prepping the data
## The targets also need to be changed to categorical format
import tensorflow.keras.utils as ku
y_train = ku.to_categorical(y_train, 18)
y_test = ku.to_categorical(y_test, 18)

######################################################################################
## Building the CNN model with two convolutional layers, maxpooling, and dropout
## Modified from lecture5_code
def get_model(numfm, numnodes, d_rate=0.5, input_shape=(50, 50, 3),
              output_size=17):

    """
    This function returns a convolution neural network Keras model,
    with numfm feature maps and numnodes neurons in the
    fully-connected layer.

    Inputs:
    - numfm: int, the number of feature maps in the convolution layer.

    - numnodes: int, the number of nodes in the fully-connected layer.

    - d_rate: float, fraction of nodes to be dropped out by the
      dropout procedure.

    - intput_shape: tuple, the shape of the input data,
    default = (50, 50, 3).

    - output_size: int, the number of nodes in the output layer,
      default = 17.

    Output: the constructed Keras model.

    """
    print("Building network.")

    ## Initialize the model.
    model = km.Sequential()

    ## Add a 2D convolution layer, with numfm feature maps.
    model.add(kl.Conv2D(numfm, kernel_size=(3, 3),
                        input_shape=input_shape,
                        activation='relu'))

    ## Add a max pooling layer.
    model.add(kl.MaxPooling2D(pool_size=(2, 2),
                              strides=(2, 2)))

    ## Add dropout to convolutional layer
    model.add(kl.Dropout(d_rate,
                         name='dropout1'))

    ## Add a 2D convolution layer, with numfm feature maps.
    model.add(kl.Conv2D(numfm, kernel_size=(3, 3),
                        input_shape=input_shape,
                        activation='relu'))

    ## Add a max pooling layer.
    model.add(kl.MaxPooling2D(pool_size=(2, 2),
                              strides=(2, 2)))

    ## Add dropout to convolutional layer
    model.add(kl.Dropout(d_rate,
                         name='dropout2'))

    ## Convert the network from 2D to 1D.
    model.add(kl.Flatten())

    ## Add a fully-connected layer.
    model.add(kl.Dense(numnodes,
                       activation='tanh'))

    ## Add the output layer.
    model.add(kl.Dense(18, activation='softmax'))

    ## Return the model.
    return model

#################################################################

## Implementing the model
model = get_model(20, 100)

# print(model.summary())

## Compiling the model
model.compile(loss = "categorical_crossentropy", optimizer = "adam",
                   metrics = ['accuracy'])

## Fiting the model
print("Training network.")
history = model.fit(datagen.flow(x_train, y_train, batch_size=32),
          steps_per_epoch=len(x_train) / 32, epochs=100)

## score of the training datasets
print("The training score is ", model.evaluate(x_train, y_train))