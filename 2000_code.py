from numpy import random
import os
import pandas as pd
import numpy as np
from keras.utils.np_utils import to_categorical
from cnn import CNN
from utils import read_data, get_training_and_test_data, read_binary_data
from constants import ROOT_DIR

method = 'hks'
path = os.path.join(ROOT_DIR, 'data', '2000-sorted-dataset')


descriptor_width = 2000
descriptor_height = 100
# for this case we need to change CATEGORICAL_CROSSENTROPY to SPARSE_CATEGOTICAL_CROSSENTROPY
def generate_arrays_from_file(method, path, descriptor_width, descriptor_height, batch_size=16):
    train = pd.read_csv(
        os.path.join(path, '{0}-train-data.csv'.format(method)))
    train.head()

    y_data = to_categorical(train.label.values)
    batch_features = np.zeros((batch_size, descriptor_width, descriptor_height, 1))
    batch_labels = np.zeros((batch_size, len(y_data[0])))

    idx = 0
    while 1:
        for _ in range(batch_size):
            i = idx % 80
            descriptor_path = os.path.join(path, method, train.filename.values[i])
            print '\n', descriptor_path
            d = np.loadtxt(descriptor_path)
            batch_features[_] = d.reshape((descriptor_width, descriptor_height, 1))
            batch_labels[_] = y_data[i]
            idx += 1
        yield (batch_features, batch_labels)



def evaluate_arrays_from_file(method, path, descriptor_width, descriptor_height):
    train = pd.read_csv(
        os.path.join(path, '{0}-test-data.csv'.format(method)))
    train.head()

    y_data = to_categorical(train.label.values)
    while 1:
        for idx, descriptor_name in enumerate(train.filename):
            descriptor_path = os.path.join(path, method, descriptor_name)
            x_data = np.loadtxt(descriptor_path)
            x_data = x_data.reshape((1, descriptor_width, descriptor_height, 1))
            yield (x_data, y_data[idx].reshape(1, 3))


cnn_model = CNN(descriptor_width=descriptor_width, descriptor_height=descriptor_height)
scores = cnn_model.train_generator(method, path, generate_arrays_from_file, evaluate_arrays_from_file, epochs=10)
print scores

'''
x_data, y_data = read_binary_data(path, descriptor_size=descriptor_width*descriptor_height)
(train_x, val_x, train_y, val_y) = get_training_and_test_data(x_data, y_data, percentage=0.6)
train_x = train_x.reshape(-1, descriptor_width, descriptor_height, 1)
val_x = val_x.reshape(-1, descriptor_width, descriptor_height, 1)

#cnn_model = CNN(descriptor_width=descriptor_width, descriptor_height=descriptor_height)
#scores = cnn_model.train(train_x, train_y, val_x, val_y, epochs=10, batch_size=32)
#print scores
'''