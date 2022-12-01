import tensorflow as tf
import numpy as np
from tensorflow import keras
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')
xs = np.arange(1, 11, dtype = float) # set up range from 1 to 10 with increments of 1 - the number of bedrooms
print(xs)
start=1
step=0.5
num=10
ys = np.arange(0,num)*step+start
print(ys)
model.fit(xs, ys, epochs=500)
