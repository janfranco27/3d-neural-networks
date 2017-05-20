import os

KP_DESCRIPTOR_ROWS = 300

KP_DESCRIPTOR_COLS = 200

KP_UNITS = KP_DESCRIPTOR_COLS * KP_DESCRIPTOR_ROWS

OUTPUT_UNITS = 30

BASE_DIR = os.path.abspath(os.path.dirname(__name__))

PLOT_DIR = os.path.join(BASE_DIR, '..', 'results', 'shrec11')

ROOT_DIR = os.path.abspath(os.path.dirname(__name__))

DATA_DIR = os.path.join(ROOT_DIR, '..', 'data')
