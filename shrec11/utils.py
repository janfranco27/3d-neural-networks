import os
import csv
import pandas as pd
import numpy as np
from keras.utils.np_utils import to_categorical
from constants import DATA_DIR


def generate_train_test_files(descriptor_dir, models_in_train=10):
    models_dir = os.path.join(DATA_DIR, descriptor_dir)
    with open(os.path.join(models_dir, 'models.csv'), 'rb') as csvfile, \
        open(os.path.join(models_dir, 'train-{0}.csv'.format(models_in_train)), 'wb') as trainfile, \
        open(os.path.join(models_dir, 'test-{0}.csv'.format(20-models_in_train)), 'wb') as testfile:

        train_writer = csv.writer(trainfile)
        train_writer.writerow(['filename','label'])
        test_writer=csv.writer(testfile)
        test_writer.writerow(['filename','label'])

        reader = csv.reader(csvfile)
        reader.next()
        for idx, row in enumerate(reader):
            if idx % 20 < models_in_train:
                train_writer.writerow(row)
            else:
                test_writer.writerow(row)


def read_data(descriptor_dir, number_of_models, method, descriptor_rows, descriptor_cols, is_training=True):

    descriptor_size = descriptor_rows * descriptor_cols
    models_dir = os.path.join(DATA_DIR, descriptor_dir)
    if is_training:
        data = pd.read_csv(os.path.join(models_dir, 'train-{0}.csv'.format(number_of_models)))
    else:
        data = pd.read_csv(os.path.join(models_dir, 'test-{0}.csv'.format(number_of_models)))

    data.head()
    y_data = to_categorical(data.label.values)

    temp = []
    for descriptor_name in data.filename:
        descriptor_path = os.path.join(models_dir, method, '{0}-{1}'.format(method, descriptor_name))
        data = np.loadtxt(descriptor_path)
        temp.append(data)

    x_data = np.stack(temp)
    x_data = x_data.reshape(-1, descriptor_size)
    return (x_data, y_data)
