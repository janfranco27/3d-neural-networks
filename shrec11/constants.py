import os

KP_DESCRIPTOR_ROWS = 100

KP_DESCRIPTOR_COLS = 100

KP_UNITS = KP_DESCRIPTOR_COLS * KP_DESCRIPTOR_ROWS

OUTPUT_UNITS = 3

BASE_DIR = os.path.abspath(os.path.dirname(__name__))

PLOT_DIR = os.path.join(BASE_DIR, '..', 'results', 'shrec11')
