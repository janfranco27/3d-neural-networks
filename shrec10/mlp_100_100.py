import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.utils import read_data, split_data
from common.mlp import MLP

from shrec10.constants import (
    KP_DESCRIPTOR_ROWS, KP_DESCRIPTOR_COLS, KP_UNITS, OUTPUT_UNITS, PLOT_DIR)

'''
method: HKS
hidden layers: 1
neurons: 10000
epochs: 12
batch_size: 16

96/96 [==============================] - 18s - loss: 1.0607 - acc: 0.5208 - val_loss: 0.9041 - val_acc: 0.5000
Epoch 2/20
96/96 [==============================] - 15s - loss: 0.6431 - acc: 0.7396 - val_loss: 0.6021 - val_acc: 0.8810
Epoch 3/20
96/96 [==============================] - 15s - loss: 0.5074 - acc: 0.8854 - val_loss: 0.5017 - val_acc: 0.9048
Epoch 4/20
96/96 [==============================] - 16s - loss: 0.4007 - acc: 0.8854 - val_loss: 0.3971 - val_acc: 0.9048
Epoch 5/20
96/96 [==============================] - 15s - loss: 0.3400 - acc: 0.8958 - val_loss: 0.3234 - val_acc: 0.9286
Epoch 6/20
96/96 [==============================] - 15s - loss: 0.2773 - acc: 0.9167 - val_loss: 0.3014 - val_acc: 0.9286
Epoch 7/20
96/96 [==============================] - 15s - loss: 0.2409 - acc: 0.9167 - val_loss: 0.2914 - val_acc: 0.9286
Epoch 8/20
96/96 [==============================] - 15s - loss: 0.2181 - acc: 0.9375 - val_loss: 0.2629 - val_acc: 0.9286
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

train_x = train_x.reshape((-1, KP_UNITS))
val_x = val_x.reshape((-1, KP_UNITS))

mlp_model = MLP(
    input_units=KP_UNITS,
    output_units=OUTPUT_UNITS,
    hidden_layers=(10000,),
    activations=('relu', 'softmax'))

scores = mlp_model.train(
    train_x, train_y, val_x, val_y, epochs=12, batch_size=16)
#mlp_model.plot(PLOT_DIR, '1-10000.png')
print scores
