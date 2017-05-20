'''
In this file the idea is to use a descriptor composed like this:

a1 a2 a3...a100 -> function R100->R, like mean, norm or max
b1 b2 b3...b100 -> function R100->R, like mean, norm or max
c1 c2 c3...c100 -> function R100->R, like mean, norm or max
.
.
.
z1 z2 z3...z100 -> function R100->R, like mean, norm or max

Then the idea is to sort the values and have a final vector-like descriptor
which is going to be used as input for a NN


NOTE:
-----
BEST CONFIGURATION HKS: 200, 150 in hidden layers
METHOD: norm
ACCURACY: 85.71 - 86.7 - 89.29


BEST CONFIGURATION WKS: 200, 150 in hidden layers
METHOD: norm
ACCURACY: 76.79 - 73.21


to get the confussion matrix:
------------------------------

val_y = val_y.argmax(1) ==> to get a 1D representation with the max index
predictions = mlp_model.model.predict(val_x)

confusion_matrix(val_y, predictions.argmax(1))
'''

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.utils import read_data, split_data, get_one_dimension_descriptor
from common.mlp import MLP

from shrec10.constants import (
    KP_DESCRIPTOR_ROWS, KP_DESCRIPTOR_COLS, OUTPUT_UNITS, PLOT_DIR)

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
x_data, y_data = read_data(
    descriptor_dir='shrec10-kp',
    method=method,
    descriptor_rows=KP_DESCRIPTOR_ROWS,
    descriptor_cols=KP_DESCRIPTOR_COLS)

(train_x, val_x, train_y, val_y) = split_data(
    x_data,
    y_data,
    split_percentage=0.7)

(train_x, val_x) = get_one_dimension_descriptor(
    'norm', KP_DESCRIPTOR_ROWS, KP_DESCRIPTOR_COLS, train_x, val_x)

mlp_model = MLP(
    input_units=KP_DESCRIPTOR_ROWS,
    output_units=OUTPUT_UNITS,
    hidden_layers=(200, 80),
    activations=('relu', 'relu', 'softmax'))

scores = mlp_model.train(
    train_x, train_y, val_x, val_y, epochs=30, batch_size=16)
print scores
