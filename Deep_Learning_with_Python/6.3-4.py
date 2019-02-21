from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

from 6.3-0 import *

model = Sequential()
model.add(layers.GRU(32,
					 input_shape = (None, floate_data.shape[-1])),
					 dropout = 0.1,
					 recurent_dropout = 0.5,
					 return_sequence = True)
model.add(layers.GRU(64,
					 activation = 'relu',
					 dropout = 0.1,
					 recurent_dropout = 0.5))
model.add(layers.Dense(1))

model.compile(optimizer = RMSprop(), loss = 'mae')
history = model.fit_generator(train_gen,
							  steps_per_epoch = 500,
							  epochs = 40,
							  validation_data = val_gen,
							  validation_steps = val_steps)
plot(hisotry)