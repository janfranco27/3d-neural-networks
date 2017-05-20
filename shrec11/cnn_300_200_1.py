import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from keras.layers import (
    Dense, Convolution2D, MaxPooling2D, Flatten, Activation, Dropout)

from common.utils import shuffle_data
from common.cnn import CNN

from shrec11.constants import (
    KP_DESCRIPTOR_ROWS, KP_DESCRIPTOR_COLS, OUTPUT_UNITS)
from shrec11.utils import read_data


method = 'hks'

train_x, train_y = read_data(
    descriptor_dir='shrec11-kp',
    number_of_models=16,
    method=method,
    descriptor_rows=KP_DESCRIPTOR_ROWS,
    descriptor_cols=KP_DESCRIPTOR_COLS)

train_x, train_y = shuffle_data(train_x, train_y)

test_x, test_y = read_data(
    descriptor_dir='shrec11-kp',
    number_of_models=4,
    method=method,
    descriptor_rows=KP_DESCRIPTOR_ROWS,
    descriptor_cols=KP_DESCRIPTOR_COLS,
    is_training=False)


train_x = train_x.reshape((-1, KP_DESCRIPTOR_ROWS, KP_DESCRIPTOR_COLS, 1))
test_x = test_x.reshape((-1, KP_DESCRIPTOR_ROWS, KP_DESCRIPTOR_COLS, 1))

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
cnn_model.model.add(Dense(300))
cnn_model.model.add(Activation('relu'))
cnn_model.model.add(Dropout(0.5))
cnn_model.model.add(Dense(OUTPUT_UNITS))
cnn_model.model.add(Activation('softmax'))

cnn_model.compile_model()

scores = cnn_model.train(train_x, train_y, test_x, test_y, epochs=20, batch_size=16)
print scores
