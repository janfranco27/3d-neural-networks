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

from sklearn.metrics import confusion_matrix
import numpy as np
from mlp import MLP
from utils import read_data, get_training_and_test_data, get_one_dimension_descriptor

method = 'hks'
x_data, y_data = read_data(method)

(train_x, val_x, train_y, val_y) = get_training_and_test_data(x_data, y_data, percentage=0.6)

(train_x, val_x) = get_one_dimension_descriptor('norm', train_x, val_x)

mlp_model = MLP(input_units=100, hidden_layers=(200, 100), activations=('relu', 'relu', 'softmax'))
scores = mlp_model.train(train_x, train_y, val_x, val_y, epochs=50, batch_size=16)

predictions = mlp_model.model.predict(val_x, batch_size=16)


print confusion_matrix(val_y.argmax(1), predictions.argmax(1))

if scores[1] > 0.83:
    print 'TRAIN', mlp_model.get_score(train_x, train_y)
    print 'VAL', mlp_model.get_score(val_x, val_y)