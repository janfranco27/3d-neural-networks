import os

ROOT_DIR = os.path.abspath('.')

DATA_DIR = os.path.join(ROOT_DIR, 'kp-dataset')

MLP_PLOT_DIR = os.path.join(ROOT_DIR, 'results/mlp/plot_loss/')

DESCRIPTOR_WIDTH = 100

DESCRIPTOR_HEIGHT = 100

DESCRIPTOR_SIZE = DESCRIPTOR_WIDTH * DESCRIPTOR_HEIGHT

OUTPUT_UNITS = 3