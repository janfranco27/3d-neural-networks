import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from keras.layers import (
    Dense, Convolution2D, MaxPooling2D, Flatten, Activation, Dropout)

from common.utils import split_data, read_multiple_channels_data
from common.cnn import CNN

from shrec10.constants import (
    KP_DESCRIPTOR_ROWS, KP_DESCRIPTOR_COLS, OUTPUT_UNITS)



method = 'hks'

x_data, y_data = read_multiple_channels_data(
    descriptor_dir='shrec10-kp',
    descriptor_rows=KP_DESCRIPTOR_ROWS,
    descriptor_cols=KP_DESCRIPTOR_COLS)

(train_x, val_x, train_y, val_y) = split_data(
    x_data,
    y_data,
    split_percentage=0.7)

train_x = train_x.reshape((-1, KP_DESCRIPTOR_ROWS, KP_DESCRIPTOR_COLS, 2))
val_x = val_x.reshape((-1, KP_DESCRIPTOR_ROWS, KP_DESCRIPTOR_COLS, 2))

cnn_model = CNN(
    descriptor_rows=KP_DESCRIPTOR_ROWS,
    descriptor_cols=KP_DESCRIPTOR_COLS,
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
cnn_model.add_layer(Dense(1500))
cnn_model.add_layer(Activation('relu'))
cnn_model.add_layer(Dense(600))
cnn_model.add_layer(Activation('relu'))
cnn_model.add_layer(Dense(OUTPUT_UNITS))
cnn_model.add_layer(Activation('softmax'))

cnn_model.compile_model()

scores = cnn_model.train(train_x, train_y, val_x, val_y, epochs=2, batch_size=16)
print scores
