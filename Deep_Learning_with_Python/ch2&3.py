"""
from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print(train_images.shape)
print(len())
"""

from keras import models 
from keras import layers

model = models.Sequential()
model.add(layers.Dense(32, activation = 'relu', input_shape = (784, )))
model.add(layers.Dense(10, activation = 'softmax'))

"""
input_tensor = layers.Input(shape = (784, ))
x = layers.Dense(32, activation = 'relu')(input_tensor)
output_tensor = layers.Dense(10, activation = 'softmax')(x)
model = models.Model(inputs = input_tensor, outputs = output_tensor)
"""
from keras import optimizers
model.compile(optimizer = optimizers.RMSprop(lr = 0.001),
			  loss = 'mse',
			  metrics = ['accuracy'])

model.fit(input_tensor, target_tensor, batch_size = 128, epochs = 10)
