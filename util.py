import re
import os
import sys
import nltk
import string

import numpy as np
import tensorflow.keras as keras

from typing import Tuple, Optional
from pathlib import Path
from sklearn.model_selection import train_test_split

from constants import *

# Global tokenizer
tokenizer = "not specified"

def load_data(train : bool) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    path = Path(DATA_BINARIES[train])

    # If the data was already prepared by another run
    if not path.exists():
        prepare_data(train=train)

    data = np.load(str(path))
    if train:
        return data['X'], data['y']
    else:
        return data['X'], None

def prepare_data(train : bool) -> None:
    global tokenizer
    if train:
        X_pos = Path(POSITIVE_TRAIN_DATA_FILE).read_text().split('\n')[:-1] # last one is empty
        X_neg = Path(NEGATIVE_TRAIN_DATA_FILE).read_text().split('\n')[:-1]
        # Remove duplicate tweets!
        X_pos = list(dict.fromkeys(X_pos))
        X_neg = list(dict.fromkeys(X_neg))
        X = X_pos + X_neg
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
def normalize_sentence(text : str) -> str:
    # lower case the text
    text = text.lower()

    # Remove whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove random numbers
    # Exclude: 4 you, 0 friends, < 3, me 2
    text = re.sub(r'\s[15-9]\s', ' ', text)
    # This needs to be done twice, find a better way
    text = re.sub(r'\s[0-9][0-9]+\s', ' ', text)
    text = re.sub(r'\s[0-9][0-9]+\s', ' ', text)

    # Remove weird non-printable characters
    text = ''.join([c for c in text if c in string.printable])

    # Specifics to the dataset
    text = re.sub(r'\( . . . \)', r' ', text)
    text = re.sub(r',', r'', text)

    # Reform Emoijis of the form (<char>) e.g. (y)
    text = re.sub(r'\(\s(?P<f1>\w)\s\)', r'(\1)', text)

    # keep common emojis
    common_emojis = [':)', ':D', ':(', ';)', ':-)', ':P', '=)', '(:',
        ';-)', ':/', 'XD', '=D', ':o', '=]', 'D:', ';D', ':]', ':-',
        '=/', '=(', '*)', ':*', '._.', ':|', '<3', '>.<', '^.^', '<3',
        '-_-', '-__-', '.__.']

    # handle emojis
    for (i, emoji) in enumerate(common_emojis):
        spaced_emoji = ' '.join(list(emoji))
        text = text.replace(emoji, ' <emoji{}> '.format(i))
        text = text.replace(spaced_emoji, ' <emoji{}> '.format(i))

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
    text = re.sub(r' \_', r' ', text)

    # Watch to not remove tokens <user>, <url>
    text = re.sub(r' \> ', r' ', text)
    text = re.sub(r' \< ', r' ', text)

    # r i p to rip -- rest of abbreviations seem fine
    text = re.sub(r'r\si\sp', r'rip', text)

    # Remove single letters apart from x, i, u, y ,z
    text = re.sub(r'\s[a-hj-tw]\s', r' ', text)
    text = re.sub(r'\s[a-hj-tw]\s', r' ', text)
    text = re.sub(r'\s[a-hj-tw]\s', r' ', text)

    # Concatenate consecutive punctuation groups
    for p in "><!?.()":
        for t in range(5):
            text = text.replace(p + ' ' + p, p * 2)

    text = nltk.WordNetLemmatizer().lemmatize(text.lower())
    return text
