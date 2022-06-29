#https://medium.com/analytics-vidhya/get-started-with-your-first-deep-learning-project-7d989cb13ae5

import numpy as np 
import matplotlib.pyplot as plt 
import random 
from keras.datasets import mnist 
from keras.models import Sequential 
from keras.layers.core import Dense, Dropout, Activation 
from keras.utils import np_utils  

#This creates 2 identical data sets. One for training and one for testing. 
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#Data Pre-Processing 
X_train = X_train.reshape(60000,784).astype('float32')
X_test = X_test.reshape(10000,784).astype('float32')  
X_train /= 255 
X_test /= 255 

#Check the shape of the data we are about to enter into a model. 
print("Training Matrix Shape: ", X_train.shape)
print("Testing Matrix Shape: ", X_test.shape)

#Building a 3-Layer Neural Network 
model = Sequential() 

#1st Layer 
model.add(Dense(512, input_shape=(784,)))
model.add(Activation('relu')) 
model.add(Dropout(0.2))  

#2nd Layer of Nodes 
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2)) 

#3rd Layer - Final layer should have nodes euqal to the number of output classes. 
model.add(Dense(10))
model.add(Activation('softmax')) 

#Summery 
model.summary() 

