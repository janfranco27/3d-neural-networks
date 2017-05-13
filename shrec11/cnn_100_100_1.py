import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from keras.layers import (
    Dense, Convolution2D, MaxPooling2D, Flatten, Activation, Dropout)

from common.utils import read_data, split_data
from common.cnn import CNN

from constants import (
    KP_DESCRIPTOR_ROWS, KP_DESCRIPTOR_COLS, OUTPUT_UNITS)



method = 'hks'

x_data, y_data = read_data(
    descriptor_dir='kp-sorted-dataset',
    method=method,
    descriptor_rows=KP_DESCRIPTOR_ROWS,
    descriptor_cols=KP_DESCRIPTOR_COLS)

(train_x, val_x, train_y, val_y) = split_data(
    x_data,
    y_data,
    split_percentage=0.7)

train_x = train_x.reshape((-1, KP_DESCRIPTOR_ROWS, KP_DESCRIPTOR_COLS, 1))
val_x = val_x.reshape((-1, KP_DESCRIPTOR_ROWS, KP_DESCRIPTOR_COLS, 1))

cnn_model = CNN(
    descriptor_rows=KP_DESCRIPTOR_ROWS,
    descriptor_cols=KP_DESCRIPTOR_COLS,
    channels=1)

nb_filters = 8
nb_pool = 2
nb_conv = 5

cnn_model.model.add(Convolution2D(64, 3, 3, input_shape=cnn_model.input_shape))
cnn_model.model.add(Activation('relu'))
cnn_model.model.add(Convolution2D(64, 3, 3))
cnn_model.model.add(Activation('relu'))
cnn_model.model.add(MaxPooling2D((2, 2)))
cnn_model.model.add(Convolution2D(64, 3, 3))
cnn_model.model.add(Activation('relu'))
cnn_model.model.add(Convolution2D(64, 3, 3))
cnn_model.model.add(Activation('relu'))
cnn_model.model.add(Dropout(0.25))
cnn_model.model.add(MaxPooling2D((2, 2)))
cnn_model.model.add(Convolution2D(128, 3, 3))
cnn_model.model.add(Activation('relu'))
cnn_model.model.add(MaxPooling2D((2, 2)))
cnn_model.model.add(Flatten())
# cnn_model.model.add(Dropout(0.2))
cnn_model.model.add(Dense(300))
cnn_model.model.add(Activation('relu'))
cnn_model.model.add(Dropout(0.5))
# cnn_model.model.add(Dense(600))
# cnn_model.model.add(Activation('relu'))
cnn_model.model.add(Dense(50))
cnn_model.model.add(Activation('softmax'))

cnn_model.compile_model()

scores = cnn_model.train(train_x, train_y, val_x, val_y, epochs=20, batch_size=16)
print scores
