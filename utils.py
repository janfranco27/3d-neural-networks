import os
import pandas as pd
import numpy as np
from keras.utils.np_utils import to_categorical
from constants import DATA_DIR, DESCRIPTOR_SIZE


def read_data(method='hks', path=DATA_DIR, descriptor_size=DESCRIPTOR_SIZE):
    train = pd.read_csv(
        os.path.join(path, '{0}-train.csv'.format(method)))
    train.head()

    temp = []
    for descriptor_name in train.filename:
        descriptor_path = os.path.join(path, method, descriptor_name)
        data = np.loadtxt(descriptor_path)
        temp.append(data)

    x_data = np.stack(temp)
    x_data = x_data.reshape(-1, descriptor_size).astype('float32')
    y_data = to_categorical(train.label.values)
    return (x_data, y_data)


def read_binary_data(path=DATA_DIR, descriptor_width=2000, descriptor_height=100):
    train = pd.read_csv(
        os.path.join(path, 'hks-train.csv'))
    train.head()

    temp = []
    for descriptor_name in train.filename:
        descriptor_path = os.path.join(path, 'hks', descriptor_name)
        data = np.loadtxt(descriptor_path)
        temp.append(data)


    train_2 = pd.read_csv(
        os.path.join(path, 'wks-train.csv'))
    train_2.head()

    temp_2 = []
    for descriptor_name in train_2.filename:
        descriptor_path = os.path.join(path, 'wks', descriptor_name)
        data = np.loadtxt(descriptor_path)
        temp_2.append(data)

    x_data = np.column_stack((temp, temp_2))
    x_data = x_data.reshape(-1, descriptor_width, descriptor_height, 2).astype('float32')
    y_data = to_categorical(train.label.values)
    return (x_data, y_data)


def get_training_and_test_data(x_data, y_data, percentage=0.75):
    seed = 128
    rng = np.random.RandomState(seed)

    split_size = int(x_data.shape[0]*percentage)

    rng_state = np.random.get_state()
    np.random.shuffle(x_data)
    np.random.set_state(rng_state)
    np.random.shuffle(y_data)

    train_x, val_x = x_data[:split_size], x_data[split_size:]
    train_y, val_y = y_data[:split_size], y_data[split_size:]
    return (train_x, val_x, train_y, val_y)


def get_training_and_test_data_no_random(x_data, y_data, split=800):
    seed = 128

    train_x, val_x = x_data[:split], x_data[split:]
    train_y, val_y = y_data[:split], y_data[split:]
    return (train_x, val_x, train_y, val_y)


one_dimension_function = {
    'mean': np.mean,
    'norm': np.linalg.norm,
    'max': np.amax
}
def get_one_dimension_descriptor(function_name, train_x, val_x):
    train_x = train_x.reshape(-1, 100, 100, 1)
    val_x = val_x.reshape(-1, 100, 100, 1)

    new_train_x = []

    for model in train_x:
    #each model has a 100x100 descriptor
        tmp = []
        for row in model:
            tmp.append(one_dimension_function[function_name](row))

        new_train_x.append(sorted(tmp))
    new_train_x = np.asarray(new_train_x)


    new_val_x = []

    for model in val_x:
        # each model has a 100x100 descriptor
        tmp = []
        for row in model:
            tmp.append(one_dimension_function[function_name](row))

        new_val_x.append(sorted(tmp))
    new_val_x = np.asarray(new_val_x)

    return new_train_x, new_val_x
