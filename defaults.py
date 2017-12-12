import numpy as np
# _USER = 'tuck'
# # _DATA_LOC = '/home/' + _USER + '/Datasets/VOC2012/'
# _DATA_LOC = '/data/dataset/traffic_line_deeplab/train/'
#
# DATA_DIRECTORY = _DATA_LOC + 'images/'
# MASK_DIRECTORY = _DATA_LOC + 'labels_tf/'
# INDEX_FILE = _DATA_LOC + 'train_tf.txt'
TRAIN_DATA_DIR = '/data/dataset/traffic_line_deeplab/train'
VAL_DATA_DIR = '/data/dataset/traffic_line_deeplab/verify'

# ATROUS_BLOCKS = 22
ATROUS_BLOCKS = 14
BATCH_SIZE = 4
IGNORE_LABEL = 255
INPUT_SIZE = (500, 500)
LEARNING_RATE = 5e-4
MOMENTUM = 0.9
NUM_CLASSES = 2
NUM_STEPS = 4000001
POWER = 0.9
RANDOM_SEED = 1234
# RESTORE_FROM = './snapshots/'
SAVE_NUM_IMAGES = 2
SAVE_MODEL_EVERY = 1000
SAVE_SUMMARY_EVERY = 25
VAL_EVERY = 20
WRITE_EVERY = 100
WRITE_FILE = 'results.txt'
# MODEL_DIR = 'recent/'
SNAPSHOT_DIR = './snapshots/recent'
SAVE_DIR = './inferenced/'
WEIGHT_DECAY = 0.0005
# bgr
# IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
# rgb
IMG_MEAN = np.array((116.66876762, 122.67891434, 104.00698793), dtype=np.float32)
