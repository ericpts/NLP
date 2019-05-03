###########################
# IMPORTS                 #
###########################
import re
import os
import nltk
import string
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras

###########################
# GLOBAL HYPERPARAMETERS  #
###########################
TRAIN_DATA_BINARY = os.path.join('datasets', 'train-data.bin.npz')
TEST_DATA_BINARY = os.path.join('datasets', 'test-data.bin.npz')
POSITIVE_TRAIN_DATA_FILE = os.path.join('datasets', 'twitter-datasets', 'train_pos_full.txt')
NEGATIVE_TRAIN_DATA_FILE = os.path.join('datasets', 'twitter-datasets', 'train_neg_full.txt')
TEST_DATA_FILE = os.path.join('datasets', 'twitter-datasets', 'test_data.txt')
DATA_BINARIES = {True: TRAIN_DATA_BINARY, False: TEST_DATA_BINARY} # not the best convention
MAX_WORDS = 20000
MAX_SEQUENCE_LENGTH = 55
PREDICTION_FILE = 'test_prediction.csv'
# Global tokenizer
tokenizer = "not specified"

def load_data(train):
    p = Path(DATA_BINARIES[train])
    # If the data was already prepared by another run
    if not p.exists():
        prepare_data(train=train)
    d = np.load(str(p))
    if train:
        return d['X'], d['y']
    else:
        # test data has no labels
        return d['X']

def prepare_data(train):
    global tokenizer
    if train:
        X_pos = Path(POSITIVE_TRAIN_DATA_FILE).read_text().split('\n')[:-1] # last one is empty
        X_neg = Path(NEGATIVE_TRAIN_DATA_FILE).read_text().split('\n')[:-1]
        # Remove duplicate tweets!
        X_pos = list(dict.fromkeys(X_pos))
        X_neg = list(dict.fromkeys(X_neg))
        X = X_pos + X_neg

        X = X[:20000]
        X = [normalize_sentence(t) for t in X]
        y = np.array([1] * len(X_pos) + [0] * len(X_neg))
        y = keras.utils.to_categorical(y, num_classes=2)
        # Allow a maximum of different words
        tokenizer = keras.preprocessing.text.Tokenizer(num_words=MAX_WORDS)
        tokenizer.fit_on_texts(X)
    else:
        X = Path(TEST_DATA_FILE).read_text().split('\n')[:-1] # Remove the index
        X = [part.split(',')[1] for part in X]
        X = [normalize_sentence(t) for t in X]
        # Bad way of doing it, TODO: save and restore the tokenizer
        if tokenizer == "not specified":
            X_pos = Path(POSITIVE_TRAIN_DATA_FILE).read_text().split('\n')[:-1] # last one is empty
            X_neg = Path(NEGATIVE_TRAIN_DATA_FILE).read_text().split('\n')[:-1]
            X_train = X_pos + X_neg
            X_train = [normalize_sentence(t) for t in X_train]
            tokenizer = keras.preprocessing.text.Tokenizer(num_words=MAX_WORDS)
            tokenizer.fit_on_texts(X_train)

    X = tokenizer.texts_to_sequences(X)
    # Pad all sentences to a fixed sequence length
    X = keras.preprocessing.sequence.pad_sequences(
            X,
            maxlen=MAX_SEQUENCE_LENGTH,
            padding='post',
            truncating='post')
    if train:
        np.savez(DATA_BINARIES[train], X=X, y=y)
    else:
        np.savez(DATA_BINARIES[train], X=X)

# Normalize a piece of text
# Tweets are whitespace separated, have <user> and <url> already
def normalize_sentence(text):
    # Remove whitespace
    text = re.sub(r'\s+', ' ', text)
    
    common_emojis = [':)', ':D', ':(', ';)', ':-)', ':P', '=)', '(:',
        ';-)', ':/', 'XD', '=D', ':o', '=]', 'D:', ';D', ':]', ':-',
        '=/', '=(', '*)', ':*', '._.', ':|', '<3', '>.<', '^.^', '<3']


    # Don't remove <3
    text = re.sub(r'[0-24-9]', '', text)
    text = re.sub(r'\< 3', '<3', text)
    # Remove weird non-printable characters
    text = ''.join([c for c in text if c in string.printable])
    # Specifics to the dataset
    text = re.sub(r' - _ - ', r' -_- ', text)
    text = re.sub(r'\( . . . \)', r' ', text)
    text = re.sub(r',', r'', text)
    # Reform Emoijis of the form (<char>)
    text = re.sub(r'\(\s(?P<f1>\w)\s\)', r'(\1)', text)
    # Remove ., _, *, {, }, ', ", |, \, :, ~,`,^,-,=
    text = re.sub(r' \. ', r' ', text)
    text = re.sub(r'\\', r'', text)
    text = re.sub(r'\s_', r' ', text)
    text = re.sub(r'\*', r'', text)
    text = re.sub(r'\{', r'', text)
    text = re.sub(r'\}', r'', text)
    text = re.sub(r'\'', r'', text)
    text = re.sub(r'\"', r'', text)
    text = re.sub(r'\|', r'', text)
    text = re.sub(r'\:', r'', text)
    text = re.sub(r'\~', r'', text)
    text = re.sub(r'\`', r'', text)
    text = re.sub(r'\^', r'', text)
    text = re.sub(r'\=', r'', text)

    # Watch to not remove -_-
    text = re.sub(r' \_', r' ', text)
    # Watch to not remove tokens <user>, <url>
    text = re.sub(r' \> ', r' ', text)
    text = re.sub(r' \< ', r' ', text)

    # r i p to rip
    text = re.sub(r'r\si\sp', r'rip', text)
    # Remove single letters apart from x
    text = re.sub(r'\s[a-wy-zA-WY-Z]\s[a-wy-zA-WY-Z]\s', r' ', text)

    text = nltk.WordNetLemmatizer().lemmatize(text.lower())
    return text
