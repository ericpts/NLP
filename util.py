import random
from pathlib import Path
import numpy as np
import swifter
import tqdm
from bs4 import BeautifulSoup
import string
import re
import pickle
import pandas as pd
import nltk
from typing import List
import string
from math import ceil
from sklearn.model_selection import train_test_split

import tensorflow as tf
import tensorflow.keras as keras

import numpy as np
from params import *

def normalize_sentence(text):
    wn = nltk.WordNetLemmatizer()
    text = BeautifulSoup(str(text), 'lxml').get_text()
    text = re.sub(r'@[a-zA-z_0-9]+', '', text)
    text = re.sub(r'http(s)?://[a-zA-Z0-9./]+', '', text)
    text = re.sub(r"'".format(string.punctuation), '', text)
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

def read_csv_data():
    cols = ['sentiment', 'id', 'date', 'query_string', 'user', 'text']
    df = pd.read_csv('./datasets/training.1600000.processed.noemoticon.csv',
            encoding='ISO-8859-1', header=None, names=cols)
    df.drop(['id', 'date', 'query_string', 'user'], axis=1, inplace=True)

    df = df.sample()

    df['text'] = df['text'].swifter.apply(normalize_sentence)

    def filter_for_nans(df):
        def is_bad_fn(t):
            return type(t) != type('str')

        # In some rows, df['text'] is float instead of string.
        # Simply drop the bad rows.
        idx = np.argwhere(df['text'].apply(is_bad_fn)).flatten()
        df.drop(labels=idx, inplace=True)

    filter_for_nans(df)

    X = list(df['text'])
    y = np.array(df['sentiment'])

    y[y == 4] = 1

    return X, y


def make_final_data(sentences, labels):
    tokenizer = keras.preprocessing.text.Tokenizer(num_words=kMaxWords)
    tokenizer.fit_on_texts(sentences)

    X = tokenizer.texts_to_sequences(sentences)
    X = keras.preprocessing.sequence.pad_sequences(
            X,
            maxlen=kMaxSequenceLength,
            padding='post',
            truncating='post')
    y = np.array(labels)
    y = keras.utils.to_categorical(y, num_classes=2)

    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.01, random_state=1337)

    np.savez('datasets/data.bin.npz',
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test)


def load_final_data():
    p = Path('datasets/data.bin.npz')
    assert p.exists()
    d = np.load(str(p))
    X_train, y_train = d['X_train'], d['y_train']
    X_test, y_test = d['X_test'], d['y_test']
    return X_train, X_test, y_train, y_test


def concatenate_data_sources(*args):
    X, y = args[0]
    for arg in args[1:]:
        X_now, y_now = arg
        X.extend(X_now)
        y = np.concatenate((y, y_now))
    return X, y


def prepare_data():
    p = Path('datasets/data.bin.npz')
    if p.exists():
        return
    X, y = concatenate_data_sources(
            read_csv_data(),
            read_eth_data()
            )

    make_final_data(X, y)

