# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 09:52:37 2020

@author: Martijn Eppenga
Example on how to create a linear regression machine learning program
Machine learning model: y = w*x + b
Data: y = w*x + b + noise
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Set A.I. parameters
eta        = 0.02
epochs     = 2
batch_size = 2

# Set model parameters
w = 0.56234
b = 12.4534

# Set number of data points
N = 3000

# Create x data
x = np.random.randn(N,1).astype(np.float32)

# Create noise
sigma = 0.1
noise = (np.random.randn(N,1)*sigma).astype(np.float32)

# Calculate y data
y = w*x + b + noise

# Create linear regression layer
class LinRegLayer(tf.keras.layers.Layer):
  def __init__(self, shape, **kwargs):
    """Creates a linear regression layer y = w*x+b
    This is effectively a dense layer without an activation:
        dense(1,use_bias=True,activation=None)"""
    super(LinRegLayer, self).__init__(**kwargs)
    # Create trainable weights and biases
    w_init = tf.random_normal_initializer()
    self.w = tf.Variable(w_init(shape), trainable=True, dtype=tf.float32)
    self.b = tf.Variable(w_init(shape), trainable=True, dtype=tf.float32)

  def call(self, Input):
    return self.w * Input + self.b


# Create model
inputs  = tf.keras.Input(shape=(1,), name="x_data")
outputs = LinRegLayer(shape=(1,))(inputs)
model   = tf.keras.Model(inputs=inputs, outputs=outputs)

# Compile model (use stochastic gradient descent for learning and squared error as loss)
optimizer = tf.keras.optimizers.SGD(learning_rate=eta)
model.compile(loss='MeanSquaredError',optimizer = optimizer)

# Train model
model.fit(x,y,batch_size=batch_size,epochs=epochs)

# Print fitted weight and bias
w_fit, b_fit  = np.array(model.get_weights())
print('Fitted weight and bias:\nw = %2.4f\nb = %2.4f' % ( w_fit, b_fit))

# Show result
plt.close()
_, ax = plt.subplots()
Fitted_model = lambda x:  w_fit*x + b_fit
x_range = np.array([min(x),max(x)])
ax.plot(x_range,Fitted_model(x_range),label='Fitted model')
# Select 100 random data points
index = (np.random.rand(100)*(len(x)-1)).astype(np.int32)
ax.plot(x[index],y[index],'o',label='Data')
ax.legend()


