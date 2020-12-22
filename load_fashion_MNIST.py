#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 14:09:38 2020

@author: kasikritdamkliang
"""

import os
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

X_train  = X_train_full[:10000] / 255.0
y_train = y_train_full[:10000]

plt.figure(figsize=(5,5))
plt.imshow(X_train[100].reshape((28, 28)), cmap = 'gray')

X_test = X_test[:3000] / 255.0
y_test = y_test[:3000]

from keras.utils import to_categorical
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)


from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Input
from keras.layers.normalization import BatchNormalization
import os
import seaborn as sns
from keras.applications.vgg16 import VGG16
SIZE = 28
VGG_model = VGG16(weights='imagenet', include_top=False, 
                  input_shape=(SIZE, SIZE, 3))

















