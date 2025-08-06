# Keras Neural Network - figures out where circles/triangles are (shown through graphical representation) 

from keras.utils import to_categorical 

from keras.models import Sequential
from keras.layers import Dense, Dropout

import numpy as np
import pandas as pd
import utils 
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')


# Sets random but reproducible seed 
np.random.seed(0) 
import tensorflow as tf 
tf.random.set_seed(1) 

# Loading Circle Dataset 

df = pd.read_csv('one_circle.csv', index_col=0)
x = np.array(df[['x_1','x_2']])
y = np.array(df['y']).astype(int)
#utils.plotPoints(x,y)

# Categorise Data - keras works better with categorised data
categorised_y = np.array(to_categorical(y,2))

# Building Model 
model = Sequential()
model.add(Dense(128, activation='relu'))
model.add(Dropout(.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(.2))
model.add(Dense(2, activation='softmax')) 

# Compiling Model 
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# Training 
model.fit(x, categorised_y, epochs=100, batch_size=10) 

# Plotting Result
utils.plotModel(x, y, model) 
