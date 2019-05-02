###########################
# IMPORTS                 #
###########################
import re
import os
import nltk
import string
from pathlib import Path
import numpy as np
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras

def normalize_sentence(text):
    wn = nltk.WordNetLemmatizer()

    text = BeautifulSoup(str(text), 'lxml').get_text()
    text = re.sub(r'@[a-zA-z_0-9]+', '', text)
    text = re.sub(r'http(s)?://[a-zA-Z0-9./]+', '', text)
    text = re.sub(r'[{}]'.format(string.punctuation), ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[0-9]+', '', text)
    text = text.lower()
    text = ''.join([c for c in text if c in string.printable])
    text = wn.lemmatize(text)

    return text


def read_eth_data():
    eth = Path('datasets/twitter-datasets/')

    txt_pos = (eth / 'train_pos_full.txt').read_text()
    txt_pos = txt_pos.split('\n')
    txt_pos = [normalize_sentence(t) for t in txt_pos]

    txt_neg = (eth / 'train_neg_full.txt').read_text()
    txt_neg = txt_neg.split('\n')
    txt_neg = [normalize_sentence(t) for t in txt_neg]

    txt = txt_pos + txt_neg
    y = [1] * len(txt_pos) + [0] * len(txt_neg)
    y = np.array(y)
    return txt, y


def make_final_data(sentences, labels):
    tokenizer = keras.preprocessing.text.Tokenizer(num_words=kMaxWords)
    tokenizer.fit_on_texts(sentences)

    X = tokenizer.texts_to_sequences(sentences)
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
MAX_SEQUENCE_LENGTH = 40

    X = keras.preprocessing.sequence.pad_sequences(
            X,
            maxlen=MAX_SEQUENCE_LENGTH,
            padding='post',
            truncating='post')
    if train:
        np.savez(DATA_BINARIES[train], X=X, y=y)
    else:
        np.savez(DATA_BINARIES[train], X=X)

