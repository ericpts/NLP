from tensorflow.keras.preprocessing.text import Tokenizer
from util import normalize_sentence
from constants import *
from pathlib import Path
from itertools import zip_longest
from typing import List


def get_tokenizer() -> Tokenizer:
    X_pos = Path(POSITIVE_TRAIN_DATA_FILE).read_text().split('\n')[:-1]
    X_neg = Path(NEGATIVE_TRAIN_DATA_FILE).read_text().split('\n')[:-1]

    # Remove duplicate tweets!
    X_pos = list(dict.fromkeys(X_pos))
    X_neg = list(dict.fromkeys(X_neg))
    X = X_pos + X_neg
    X = [normalize_sentence(t) for t in X]

    tokenizer = Tokenizer(num_words=MAX_WORDS)
    tokenizer.fit_on_texts(X)

    return tokenizer


def process(text: str) -> List[str]:
    lines = []

    words = text.split(' ')
    line = ""
    for word in words:
        if len(line + ' ' + word) > 80:
            lines.append(line)
            line   = ''
        line += ' ' + word
    if len(line) > 0:
        lines.append(line)
    return lines


if __name__ == "__main__":
    tokenizer = get_tokenizer()

    X  = Path(TEST_DATA_FILE).read_text().split('\n')[:-1]
    X0 = X

    X = [','.join(part.split(',')[1:]) for part in X]
    X = [normalize_sentence(t) for t in X]
    X = tokenizer.sequences_to_texts(tokenizer.texts_to_sequences(X))

    print("{:80}   |  {:80}".format("Before", "After"))
    print("-"*180)
    part = 100
    for x, y in zip(X0[:part], X[:part]):
        xs, ys = process(x), process(y)
        xs, ys = xs + [''], ys + ['']
        for x, y in zip_longest(xs, ys, fillvalue=''):
            print("{:80}   |  {:80}".format(x, y))
