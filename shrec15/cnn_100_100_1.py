import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from keras.layers import (
    Dense, Convolution2D, MaxPooling2D, Flatten, Activation, Dropout)

from common.utils import read_data, get_training_and_test_data_no_random
from common.cnn import CNN

from shrec15.constants import (
    KP_DESCRIPTOR_ROWS, DESCRIPTOR_COLS, OUTPUT_UNITS, SPLIT_SIZE)



method = 'hks'

#x_data, y_data = read_data(
#    descriptor_dir='shrec-15-kp',
#    method=method,
#    descriptor_rows=KP_DESCRIPTOR_ROWS,
#    descriptor_cols=DESCRIPTOR_COLS)
#
#(train_x, val_x, train_y, val_y) = get_training_and_test_data_no_random(
#    x_data,
#    y_data,
#    split=SPLIT_SIZE
#)
#
#train_x = train_x.reshape((-1, KP_DESCRIPTOR_ROWS, DESCRIPTOR_COLS, 1))
#val_x = val_x.reshape((-1, KP_DESCRIPTOR_ROWS, DESCRIPTOR_COLS, 1))

cnn_model = CNN(
    descriptor_rows=KP_DESCRIPTOR_ROWS,
    descriptor_cols=DESCRIPTOR_COLS,
    channels=1)

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
cnn_model.add_layer(Dense(1500))
cnn_model.add_layer(Activation('relu'))
cnn_model.add_layer(Dense(600))
cnn_model.add_layer(Activation('relu'))
cnn_model.add_layer(Dense(OUTPUT_UNITS))
cnn_model.add_layer(Activation('softmax'))

cnn_model.compile_model()

scores = cnn_model.train_generator(
    dir='shrec-15-kp', method='hks', rows= KP_DESCRIPTOR_ROWS, cols=100,
    training_size=10, validation_size=2,epochs=2, batch_size=2)
