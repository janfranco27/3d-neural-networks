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
        os.path.join(models_dir, 'models.csv'.format(method)))
    train.head()

    temp = []
    for descriptor_name in train.filename:
        descriptor_path = os.path.join(models_dir, method, '{0}-{1}'.format(method, descriptor_name))
        data = np.loadtxt(descriptor_path)
        temp.append(data)

    x_data = np.stack(temp)
    x_data = x_data.reshape(-1, descriptor_size)
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
            os.path.join(models_dir, 'models.csv'.format(method)))
        train.head()

        y_data = to_categorical(train.label.values)

        temp = []
        for descriptor_name in train.filename:
            descriptor_path = os.path.join(models_dir, method, '{0}-{1}'.format(method, descriptor_name))
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


def train_with_generator(descriptor_dir, descriptor_rows,
                         descriptor_cols=1, training_set_size=800,
                         batch_size=32, channels=1, method='hks'):

    models_dir = os.path.join(DATA_DIR, descriptor_dir)
    train = pd.read_csv(
        os.path.join(models_dir, '{0}-train.csv'.format(method)))
    train.head()

    y_data = to_categorical(train.label.values)
    if not descriptor_cols == 1:
        batch_features = np.zeros((
            batch_size, descriptor_rows, descriptor_cols, channels))
    else:
        batch_features = np.zeros(
            (batch_size, descriptor_rows))
    batch_labels = np.zeros((batch_size, len(y_data[0])))

    idx = 0
    while 1:
        for _ in range(batch_size):
            i = idx % training_set_size
            if channels == 1:
                batch_features[_] = read_one_channel(models_dir,
                                                     method,
                                                     train.filename.values[i],
                                                     descriptor_rows,
                                                     descriptor_cols)
            else:
                batch_features[_] = read_two_channels(models_dir,
                                                     train.filename.values[i],
                                                     descriptor_rows,
                                                     descriptor_cols)
            batch_labels[_] = y_data[i]
            idx += 1
        yield (batch_features, batch_labels)


def read_one_channel(models_dir, method, filename, rows, cols=1, is_training=False):
    descriptor_path = os.path.join(models_dir, method, filename)
    d = np.loadtxt(descriptor_path)

    if cols is not 1:
        d = d.reshape((rows, cols, 1))
    else:
        d = d.reshape((rows))

    if is_training:
        if not cols == 1:
            d = d.reshape(1, rows, cols, 1)
        else:
            d = d.reshape(1, rows)
    return d


def read_two_channels(models_dir, filename, rows, cols, is_training=False):
    methods = ('hks', 'wks')
    filename = filename[4:]
    temp = []
    for method in methods:
        descriptor_path = os.path.join(models_dir,
                                       method,
                                       '{0}-{1}'.format(method, filename))
        data = np.loadtxt(descriptor_path)
        temp.append(data)

    x_data = np.column_stack(temp)
    x_data = x_data.reshape(rows, cols, len(methods))

    if is_training:
        if not cols == 1:
            x_data = x_data.reshape(1, rows, cols, len(methods))
        else:
            x_data = x_data.reshape(1, rows, len(methods))

    return x_data


def evaluate_with_generator(descriptor_dir, descriptor_rows, descriptor_cols=1,
                            channels=1, method='hks'):

    models_dir = os.path.join(DATA_DIR, descriptor_dir)
    train = pd.read_csv(
        os.path.join(models_dir, '{0}-test.csv'.format(method)))
    train.head()

    y_data = to_categorical(train.label.values)
    while 1:
        for idx, descriptor_name in enumerate(train.filename):

            if channels == 1:
                x_data = read_one_channel(models_dir,
                                          method,
                                          descriptor_name,
                                          descriptor_rows,
                                          descriptor_cols, is_training=True)
            else:
                x_data = read_two_channels(models_dir,
                                           descriptor_name,
                                           descriptor_rows,
                                           descriptor_cols, is_training=True)

            yield (x_data, y_data[idx].reshape(1, len(y_data[0])))
