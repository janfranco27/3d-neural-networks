import os
import pandas as pd
import numpy as np
from keras.utils.np_utils import to_categorical
from constants import DATA_DIR


def read_data(descriptor_dir, method, descriptor_rows, descriptor_cols):
    '''
    Parameters
    ----------
    method: hks or wks
    descriptor_size: descriptor_rows * descriptor_cols
    descriptor_dir: folder name of the descriptor
    '''

    descriptor_size = descriptor_rows * descriptor_cols
    models_dir = os.path.join(DATA_DIR, descriptor_dir)
    train = pd.read_csv(
        os.path.join(models_dir, '{0}-train.csv'.format(method)))
    train.head()

    temp = []
    for descriptor_name in train.filename:
        descriptor_path = os.path.join(models_dir, method, descriptor_name)
        data = np.loadtxt(descriptor_path)
        temp.append(data)

    x_data = np.stack(temp)
    x_data = x_data.reshape(-1, descriptor_size).astype('float32')
    y_data = to_categorical(train.label.values)
    return (x_data, y_data)


def read_multiple_channels_data(descriptor_dir, descriptor_rows,
                                descriptor_cols, methods=('hks', 'wks')):
    '''
    The files hks-train and wks-train must have the models in the same order
    because the train labels are override
    '''
    models_dir = os.path.join(DATA_DIR, descriptor_dir)

    data_loaded = []
    y_data = []
    for method in methods:
        train = pd.read_csv(
            os.path.join(models_dir, '{0}-train.csv'.format(method)))
        train.head()

        y_data = to_categorical(train.label.values)

        temp = []
        for descriptor_name in train.filename:
            descriptor_path = os.path.join(models_dir, method, descriptor_name)
            data = np.loadtxt(descriptor_path)
            temp.append(data)
        data_loaded.append(temp)

    x_data = np.column_stack(data_loaded)
    x_data = x_data.reshape(
        -1, descriptor_rows, descriptor_cols, 2).astype('float32')
    return (x_data, y_data)


def shuffle_data(x_data, y_data):
    seed = 128
    rng = np.random.RandomState(seed)

    rng_state = np.random.get_state()
    np.random.shuffle(x_data)
    np.random.set_state(rng_state)
    np.random.shuffle(y_data)
    return (x_data, y_data)


def split_data(x_data, y_data, split_percentage=0.75):
    split_size = int(x_data.shape[0] * split_percentage)

    x_data, y_data = shuffle_data(x_data, y_data)
    train_x, val_x = x_data[:split_size], x_data[split_size:]
    train_y, val_y = y_data[:split_size], y_data[split_size:]
    return (train_x, val_x, train_y, val_y)


def get_training_and_test_data_no_random(x_data, y_data, split=800):
    train_x, val_x = x_data[:split], x_data[split:]
    train_y, val_y = y_data[:split], y_data[split:]
    return (train_x, val_x, train_y, val_y)


one_dimension_function = {
    'mean': np.mean,
    'norm': np.linalg.norm,
    'max': np.amax
}


def _get_one_dimension_descriptor_for_data(function_name, data_x):
    new_data_x = []
    for model in data_x:
        tmp = []
        for row in model:
            tmp.append(one_dimension_function[function_name](row))

        new_data_x.append(sorted(tmp))
    return np.asarray(new_data_x)


def get_one_dimension_descriptor(function_name, descriptor_rows,
                                 descriptor_cols, train_x, val_x):

    train_x = train_x.reshape(-1, descriptor_rows, descriptor_cols, 1)
    val_x = val_x.reshape(-1, descriptor_rows, descriptor_cols, 1)

    new_train_x = _get_one_dimension_descriptor_for_data(function_name, train_x)

    new_val_x = _get_one_dimension_descriptor_for_data(function_name, val_x)

    return new_train_x, new_val_x
