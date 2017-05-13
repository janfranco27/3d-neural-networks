import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.mlp import MLP

from shrec15.constants import (
    KP_DESCRIPTOR_ROWS, DESCRIPTOR_COLS, OUTPUT_UNITS, SPLIT_SIZE)

method = 'hks'

mlp_model = MLP(
    input_units=KP_DESCRIPTOR_ROWS * 100,
    output_units=OUTPUT_UNITS,
    hidden_layers=(200, 100),
    activations=('relu', 'relu', 'softmax'))

scores = mlp_model.train_generator(
    dir='shrec15-kp', channels=1, method='hks', rows= KP_DESCRIPTOR_ROWS * 100, cols=1,
    training_size=10, validation_size=2,epochs=2, batch_size=2)
