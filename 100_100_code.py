from mlp import MLP
from cnn import CNN
from utils import read_binary_data, get_training_and_test_data
import numpy as np

x_data, y_data = read_binary_data('kp-sorted-dataset', 100, 100)
(train_x, val_x, train_y, val_y) = get_training_and_test_data(x_data, y_data, percentage=0.6)

cnn_model = CNN(descriptor_width=100, descriptor_height=100)
scores = cnn_model.train(train_x, train_y, val_x, val_y, epochs=20, batch_size=16)
print scores


'''
cnn tried:
---------

self.model.add(Convolution2D(8, 5, 5))
self.model.add(Activation('relu'))
self.model.add(MaxPooling2D(2, 2))
self.model.add(Convolution2D(16, 5, 5))
self.model.add(Activation('relu'))
self.model.add(MaxPooling2D(4, 4))
self.model.add(Flatten())
self.model.add(Dense(3500))
self.model.add(Activation('relu'))
self.model.add(Dropout(0.2))
self.model.add(Dense(1500))
self.model.add(Activation('relu'))
self.model.add(Dense(3))
self.model.add(Activation('softmax'))




cnn_model = CNN()
scores = cnn_model.train(train_x, train_y, val_x, val_y, epochs=20, batch_size=16)
print scores
'''


'''

#---------------------------------------------------------------------------
#---------------------------------------------------------------------------
# reshape data


# define vars
input_shape = (100*100,)
input_reshape = (100, 100, 1)

conv_num_filters = 5
conv_filter_size = 10

pool_size = (3, 3)

hidden_num_units = 1000
output_num_units = 3

epochs = 5
batch_size = 10

from keras.layers import InputLayer, Convolution2D, MaxPooling2D, Flatten

model = Sequential([
 InputLayer(input_shape=input_reshape),

 Convolution2D(10, 5, 5, activation='relu'),
 MaxPooling2D(pool_size=pool_size),
 Flatten(),
 Dense(output_dim=hidden_num_units, activation='relu'),
 Dense(output_dim=output_num_units, input_dim=hidden_num_units, activation='softmax'),
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

trained_model_conv = model.fit(train_x_temp, train_y, nb_epoch=epochs, batch_size=batch_size, validation_data=(val_x_temp, val_y))
'''