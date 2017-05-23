import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from keras.layers import (
    Dense, Convolution2D, MaxPooling2D, Flatten, Activation, Dropout, InputLayer, BatchNormalization)

from common.utils import read_data, get_training_and_test_data_no_random
from common.cnn import CNN

from shrec11.constants import OUTPUT_UNITS



cnn_model = CNN(
    descriptor_rows=50,
    descriptor_cols=50,
    channels=2)

nb_filters = 8
nb_pool = 2
nb_conv = 5

cnn_model.add_layer(Convolution2D(64, nb_conv, nb_conv, input_shape=cnn_model.input_shape))
cnn_model.add_layer(Activation('relu'))
cnn_model.add_layer(MaxPooling2D((nb_pool, nb_pool)))
cnn_model.add_layer(Convolution2D(64, 4, 4))
cnn_model.add_layer(Activation('relu'))
cnn_model.add_layer(MaxPooling2D((nb_pool*2, nb_pool*2)))
cnn_model.add_layer(Convolution2D(32, 3, 3))
cnn_model.add_layer(Activation('relu'))
cnn_model.add_layer(Flatten())
cnn_model.add_layer(Dropout(0.25))
cnn_model.add_layer(Dense(4000))
cnn_model.add_layer(Activation('relu'))
cnn_model.add_layer(Dense(OUTPUT_UNITS))
cnn_model.add_layer(Activation('softmax'))

cnn_model.compile_model()

scores = cnn_model.train_generator(
    dir='shrec11-kp', channels=2, rows=50, cols=50,
    training_size=480, validation_size=120,epochs=30, batch_size=40,
    method=('hks-50', 'wks-50'))
