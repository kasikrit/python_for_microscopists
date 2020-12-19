#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 14:52:39 2020

@author: kasikritdamkliang
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from tensorflow import keras
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm

print(tf.__version__)
print(keras.__version__)


fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

print(X_train_full.shape)

print(X_train_full.dtype)

# For faster training, let's use a subset 10,000
X_train, y_train = X_train_full[:10000] / 255.0, y_train_full[:10000]

print(X_train.shape, y_train.shape)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

def create_model(): 
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(300, activation='relu'),
        Dense(100, activation='relu'),
        Dense(10, activation='softmax'),
    ])
    return model


from tensorflow.keras.callbacks import LearningRateScheduler
import tensorflow_addons as tfa
from keras.optimizers import SGD

epochs=100
learning_rate = 0.1
decay_rate = learning_rate / epochs
momentum = 0.8

#Default values for SGD. lr=0.1, m=0, decay=0
#Nesterov momentum is a different version of the momentum method.
#Nesterov has stronger theoretical converge guarantees for convex functions.
sgd = SGD(lr=learning_rate, momentum=momentum, 
          decay=decay_rate, nesterov=False)

# Create a new model
lr_exp_model = create_model()

lr_exp_model.compile(
    optimizer=sgd,
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)

lr_exp_model.summary()


def lr_exp_decay(epoch):
    lrate = learning_rate * np.exp(-decay_rate*epoch)
    return lrate

tqdm_callback = tfa.callbacks.TQDMProgressBar(show_epoch_progress=False)

# Fit the model to the training data
history_lr_exp_decay = lr_exp_model.fit(
    X_train, 
    y_train, 
    epochs=epochs, 
    validation_split=0.2,
    batch_size=32,
    callbacks=[LearningRateScheduler(lr_exp_decay, verbose=1),
               tqdm_callback]
)







