import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from keras.layers import (
    Dense, Convolution2D, MaxPooling2D, Flatten, Activation, Dropout, InputLayer, BatchNormalization)

from common.utils import read_data, get_training_and_test_data_no_random
from common.cnn import CNN

from shrec11.constants import OUTPUT_UNITS



cnn_model = CNN(
    descriptor_rows=64,
    descriptor_cols=64,
    channels=2)

nb_filters = 8
nb_pool = 2
nb_conv = 5

cnn_model.add_layer(Convolution2D(8, nb_conv, nb_conv, input_shape=cnn_model.input_shape))
cnn_model.add_layer(Activation('relu'))
cnn_model.add_layer(MaxPooling2D((nb_pool*2, nb_pool*2)))
cnn_model.add_layer(Convolution2D(12, 4, 4))
cnn_model.add_layer(Activation('relu'))
cnn_model.add_layer(MaxPooling2D((nb_pool*2, nb_pool*2)))
cnn_model.add_layer(Convolution2D(16, 3, 3))
cnn_model.add_layer(Activation('relu'))
cnn_model.add_layer(Flatten())
cnn_model.add_layer(Dropout(0.2))
cnn_model.add_layer(Dense(512))
cnn_model.add_layer(Activation('relu'))
cnn_model.add_layer(Dense(OUTPUT_UNITS))
cnn_model.add_layer(Activation('softmax'))

cnn_model.compile_model()

scores = cnn_model.train_generator(
    dir='shrec11-kp', channels=2, rows=64, cols=64,
    training_size=480, validation_size=120,epochs=30, batch_size=40,
    method=('hks-64', 'wks-64'))



'''
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from keras.layers import (
    Dense, Convolution2D, MaxPooling2D, Flatten, Activation, Dropout)

from common.utils import read_data, get_training_and_test_data_no_random
from common.cnn import CNN

from shrec11.constants import OUTPUT_UNITS



cnn_model = CNN(
    descriptor_rows=300,
    descriptor_cols=200,
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
cnn_model.add_layer(Dense(2000))
cnn_model.add_layer(Activation('relu'))
cnn_model.add_layer(Dense(500))
cnn_model.add_layer(Activation('relu'))
cnn_model.add_layer(Dense(OUTPUT_UNITS))
cnn_model.add_layer(Activation('softmax'))

cnn_model.compile_model()

scores = cnn_model.train_generator(
    dir='shrec11-kp', channels=2, rows=300, cols=200,
    training_size=16*30, validation_size=4*30,epochs=10, batch_size=48,
    method=('hks', 'wks'))
'''