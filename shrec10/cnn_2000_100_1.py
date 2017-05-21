import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from keras.layers import (
    Dense, Convolution2D, MaxPooling2D, Flatten, Activation, Dropout)

from common.utils import read_data, get_training_and_test_data_no_random
from common.cnn import CNN

from shrec10.constants import OUTPUT_UNITS



method = 'hks'

cnn_model = CNN(
    descriptor_rows=2000,
    descriptor_cols=100,
    channels=2)

nb_filters = 8
nb_pool = 2
nb_conv = 5

cnn_model.add_layer(Convolution2D(
    nb_filters*4, nb_conv, nb_conv, input_shape=cnn_model.input_shape))
cnn_model.add_layer(Activation('relu'))
cnn_model.add_layer(MaxPooling2D((nb_pool, nb_pool)))
cnn_model.add_layer(Convolution2D(nb_filters*4, nb_conv, nb_conv))
cnn_model.add_layer(Activation('relu'))
cnn_model.add_layer(MaxPooling2D((nb_pool, nb_pool)))
cnn_model.add_layer(Flatten())
cnn_model.add_layer(Dropout(0.2))
cnn_model.add_layer(Dense(10))
cnn_model.add_layer(Activation('relu'))
cnn_model.add_layer(Dense(6))
cnn_model.add_layer(Activation('relu'))
cnn_model.add_layer(Dense(OUTPUT_UNITS))
cnn_model.add_layer(Activation('softmax'))

cnn_model.compile_model()

scores = cnn_model.train_generator(
    dir='shrec10-qslim', channels=2, rows=2000, cols=100,
    training_size=100, validation_size=20,epochs=2, batch_size=20,
    method=('hks', 'wks'))
