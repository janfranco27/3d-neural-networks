import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.utils import read_data, get_training_and_test_data_no_random
from common.mlp import MLP

from shrec15.constants import (
    KP_DESCRIPTOR_ROWS, DESCRIPTOR_COLS, OUTPUT_UNITS, SPLIT_SIZE)


method = 'hks'
x_data, y_data = read_data(
    descriptor_dir='shrec15-kp',
    method=method,
    descriptor_rows=KP_DESCRIPTOR_ROWS,
    descriptor_cols=DESCRIPTOR_COLS)


(train_x, val_x, train_y, val_y) = get_training_and_test_data_no_random(
    x_data,
    y_data,
    split=SPLIT_SIZE
)


mlp_model = MLP(
    input_units=KP_DESCRIPTOR_ROWS * 100,
    output_units=OUTPUT_UNITS,
    hidden_layers=(100,),
    activations=('relu', 'softmax'))

scores = mlp_model.train(
    train_x, train_y, val_x, val_y, epochs=4, batch_size=50)
#mlp_model.plot(PLOT_DIR, '1-10000.png')
print scores
