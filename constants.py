import os


TRAIN_DATA_BINARY = os.path.join('datasets', 'train-data.bin.npz')
TEST_DATA_BINARY = os.path.join('datasets', 'test-data.bin.npz')
POSITIVE_TRAIN_DATA_FILE = os.path.join('datasets', 'twitter-datasets', 'train_pos_full.txt')
NEGATIVE_TRAIN_DATA_FILE = os.path.join('datasets', 'twitter-datasets', 'train_neg_full.txt')
TEST_DATA_FILE = os.path.join('datasets', 'twitter-datasets', 'test_data.txt')

DATA_BINARIES = {
    True: TRAIN_DATA_BINARY,
    False: TEST_DATA_BINARY
}

PREDICTION_FILE = 'test_prediction.csv'


MAX_WORDS = 20000
MAX_SEQUENCE_LENGTH = 55
