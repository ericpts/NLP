#!/usr/bin/env python3
import random
from pathlib import Path
import numpy as np
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

import tensorflow as tf
import tensorflow.keras as keras

import numpy as np

from util import *


def twitter_model():
    inputs = keras.Input(shape=(kMaxSequenceLength, ))

    X = inputs
    X = keras.layers.Embedding(kMaxWords, 128, input_length=kMaxSequenceLength)(X)
    X = keras.layers.Dropout(1 / 4)(X)
    X = keras.layers.Conv1D(64, 5, strides=1, padding='valid', activation='relu')(X)
    X = keras.layers.LSTM(64)(X)
    X = keras.layers.Dense(2)(X)

    model = keras.models.Model(inputs=inputs, outputs=X, name='TwitterModel')

    model.summary()
    return model

def main():
    prepare_data()

    X_train, X_test, y_train, y_test = load_final_data()

    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]

    print('Training on {}, validating on {}'.format(X_train.shape[0], X_test.shape[0]))


    if Path('model.bin').exists():
        model = keras.models.load_model('model.bin')
        print('Loaded model from disk.')
    else:
        model = twitter_model()

    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, validation_data=(X_test, y_test))

    model.save('model.bin')


if __name__ == '__main__':
    main()
