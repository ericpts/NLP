import os

TRAIN_DATA_BINARY = os.path.join('datasets', 'train-data.bin.npz')
TEST_DATA_BINARY = os.path.join('datasets', 'test-data.bin.npz')
POSITIVE_TRAIN_DATA_FILE = os.path.join('datasets', 'twitter-datasets', 'train_pos_full.txt')
NEGATIVE_TRAIN_DATA_FILE = os.path.join('datasets', 'twitter-datasets', 'train_neg_full.txt')
TEST_DATA_FILE = os.path.join('datasets', 'twitter-datasets', 'test_data.txt')

TRAIN_DATA_TEXT = os.path.join('datasets', 'text-train-data.bin.npz')
TEST_DATA_TEXT = os.path.join('datasets', 'text-test-data.bin.npz')

DATA_BINARIES = {
    True: TRAIN_DATA_BINARY,
    False: TEST_DATA_BINARY
}

DATA_TEXT = {
    True: TRAIN_DATA_TEXT,
    False: TEST_DATA_TEXT
}

PREDICTION_FILE = 'test_prediction.csv'

TRAIN_TEST_SPLIT_PERCENTAGE = 0.1

# Used for assigning numbers to most popular words and ignoring the rest
MAX_WORDS = 20000
# Used for padding the smaller tweets
MAX_SEQUENCE_LENGTH = 50
# Arguments provided to the main function
ARGS = None
# Embedding dimension of words
EMBEDDING_DIM = 100
