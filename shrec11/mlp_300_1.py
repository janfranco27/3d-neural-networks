import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from keras.layers import Dense
from keras.layers.normalization import BatchNormalization

from common.mlp import MLP
from common.utils import get_one_dimension_descriptor, shuffle_data

from shrec11.utils import read_data
from shrec11.constants import (
    KP_DESCRIPTOR_ROWS, KP_DESCRIPTOR_COLS, OUTPUT_UNITS)

method = 'hks'
train_x, train_y = read_data(
    descriptor_dir='shrec11-kp',
    number_of_models=10,
    method=method,
    descriptor_rows=KP_DESCRIPTOR_ROWS,
    descriptor_cols=KP_DESCRIPTOR_COLS)

train_x, train_y = shuffle_data(train_x, train_y)

test_x, test_y = read_data(
    descriptor_dir='shrec11-kp',
    number_of_models=10,
    method=method,
    descriptor_rows=KP_DESCRIPTOR_ROWS,
    descriptor_cols=KP_DESCRIPTOR_COLS,
    is_training=False)

(train_x, test_x) = get_one_dimension_descriptor(
    'norm', KP_DESCRIPTOR_ROWS, KP_DESCRIPTOR_COLS, train_x, test_x)

mlp_model = MLP(
    input_units=KP_DESCRIPTOR_ROWS,
    output_units=OUTPUT_UNITS)

mlp_model.add_layer(Dense(output_dim=400, input_dim=KP_DESCRIPTOR_ROWS, activation='relu'))
mlp_model.add_layer(BatchNormalization())
mlp_model.add_layer(Dense(output_dim=100, input_dim=400, activation='relu'))
mlp_model.add_layer(Dense(output_dim=OUTPUT_UNITS, input_dim=100, activation='softmax'))
mlp_model.compile_model()

scores = mlp_model.train(
    train_x, train_y, test_x, test_y, epochs=150, batch_size=48)

print scores
