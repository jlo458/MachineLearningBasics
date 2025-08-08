# Number Recognition 

from keras.utils import to_categorical 

from keras.models import Sequential
from keras.layers import Dense, Dropout

import numpy as np
import pandas as pd
import utils 
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')

from keras import datasets


# Loading Data from MNIST
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data() 


# Reshaping labels into singular vectors
x_train_reshaped = x_train.reshape(-1, 28*28)
x_test_reshaped = x_test.reshape(-1, 28*28) 


# Categorising Data
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)


# Making Model
model = Sequential()
model.add(Dense(128, activation='relu'))
model.add(Dropout(.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(.2))
model.add(Dense(10, activation='softmax'))

# Compiling Model 
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# Training 
model.fit(x_train_reshaped, y_train_cat, epochs=10, batch_size=10) 

# Evaluation 
pred_vector = model.predict(x_test_reshaped)
preds = [np.argmax(pred) for pred in pred_vector]
