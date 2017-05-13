import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.utils import (
    read_data, get_training_and_test_data_no_random,
    get_one_dimension_descriptor)
from common.mlp import MLP

from shrec15.constants import (
    KP_DESCRIPTOR_ROWS, DESCRIPTOR_COLS, OUTPUT_UNITS, SPLIT_SIZE)

'''
method: HKS
one_dimension_descriptor: norm
hidden layers: 2
neurons: 200, 80
epochs: 30
batch_size: 16

Epoch 26/30
96/96 [==============================] - 0s - loss: 0.4773 - acc: 0.8542 - val_loss: 0.5186 - val_acc: 0.8571
Epoch 27/30
96/96 [==============================] - 0s - loss: 0.4685 - acc: 0.8542 - val_loss: 0.5120 - val_acc: 0.8571
Epoch 28/30
96/96 [==============================] - 0s - loss: 0.4615 - acc: 0.8542 - val_loss: 0.5191 - val_acc: 0.8571
Epoch 29/30
96/96 [==============================] - 0s - loss: 0.4365 - acc: 0.8646 - val_loss: 0.4968 - val_acc: 0.8571
Epoch 30/30
96/96 [==============================] - 0s - loss: 0.4340 - acc: 0.8542 - val_loss: 0.4854 - val_acc: 0.8571
'''

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

#(train_x, train_y) = shuffle_data(train_x, train_y)

#(val_x, val_y) = shuffle_data(val_x, val_y)

#(train_x, val_x) = get_one_dimension_descriptor(
#    'norm', KP_DESCRIPTOR_ROWS, DESCRIPTOR_COLS, train_x, val_x)

#train_x = train_x.reshape(-1, 10000)
#val_x = val_x.reshape(-1, 10000)


mlp_model = MLP(
    input_units=KP_DESCRIPTOR_ROWS * 100,
    output_units=OUTPUT_UNITS,
    hidden_layers=(10000, 4000),
    activations=('relu', 'relu', 'softmax'))

scores = mlp_model.train_generator(
    dir='shrec-15-kp', method='hks', rows= KP_DESCRIPTOR_ROWS * 100, cols=1,
    training_size=10, validation_size=2,epochs=2, batch_size=2)
