import re
import nltk
import string
import pickle
import numpy as np
import tensorflow.keras as keras


from typing import Tuple, Optional
from pathlib import Path
from constants import *


def save_object(obj : object, name : str) -> None:
    '''
        Saves 'obj' object in the OBJECT_DIRECTORY with the provided 'name'
        Args:
        obj            Object to be saved
        name           Name to be given to the file
    '''
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_object(name : str) -> object:
    '''
        Loads object with the provided 'name' from the OBJECT_DIRECTORY
        Args:
        name           Name of object to be loaded
    '''
    with open(name, 'rb') as f:
        return pickle.load(f)

def load_data(
    train : bool,
    as_text: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    '''
    train: Whether we want the training data.
    as_text:
        True if we should return the data as a list of strings.
        False if the data should be returned as a list of integers, where each
            integer uniquely identifies a token.
    '''
    path = Path(DATA_TEXT[train]) if as_text else Path(DATA_BINARIES[train])

    # If the data was already prepared by another run
    if not path.exists():
        prepare_data(train=train, as_text=as_text)

    data = np.load(str(path))
    if train:
        return data['X'], data['y']
    else:
        return data['X'], None


def prepare_data(train: bool, as_text: bool) -> None:
    tokenizer = None
    if Path(TOKENIZER_PATH).is_file():
        tokenizer = load_object(TOKENIZER_PATH)

    if not Path(TOKENIZER_PATH).is_file() or train:
        X_pos = Path(POSITIVE_TRAIN_DATA_FILE).read_text().split('\n')[:-1]
        X_neg = Path(NEGATIVE_TRAIN_DATA_FILE).read_text().split('\n')[:-1]

        # Remove duplicate tweets!
        X_pos = list(dict.fromkeys(X_pos))
        X_neg = list(dict.fromkeys(X_neg))
        X = X_pos + X_neg
        X = [normalize_sentence(t) for t in X]
        y = np.array([1] * len(X_pos) + [0] * len(X_neg))

        # Allow a maximum of different words
        tokenizer = keras.preprocessing.text.Tokenizer(num_words=MAX_WORDS)
        tokenizer.fit_on_texts(X)

        # Save tokenizer
        save_object(tokenizer, TOKENIZER_PATH)

    if not train:
        X = Path(TEST_DATA_FILE).read_text().split('\n')[:-1]
        X = [','.join(part.split(',')[1:]) for part in X]
        X = [normalize_sentence(t) for t in X]

    # Saving processed text
    if as_text:
        if C['ELMO_SEQ']:
            def supplement(l):
                l2 = ['' for i in range(MAX_SEQUENCE_LENGTH - len(l))]
                return l + l2
            X = [supplement(x.split()) for x in X]
            print(len(X[0]))
        if train:
            np.savez(DATA_TEXT[train], X=X, y=y)
        else:
            np.savez(DATA_TEXT[train], X=X)
        return

    # At this point, the text is normalized and the tokenizer is loaded
    X = tokenizer.texts_to_sequences(X)

    # Pad all sentences to a fixed sequence length
    X = keras.preprocessing.sequence.pad_sequences(
        sequences=X,
        maxlen=MAX_SEQUENCE_LENGTH,
        padding='post',
    )
    if train:
        np.savez(DATA_BINARIES[train], X=X, y=y)
    else:
        np.savez(DATA_BINARIES[train], X=X)


def handle_emojis(text : str) -> str:
    # Translate common emojis to words to help the model
    emoji_dictionary = {
        'happy': [':)', ':D', ';)', ':-)', ':P', '=)', '(:',
        ';-)', '=D', '=]', ';D', ':]', '^.^', '(y)'],
        'sad': [':(', ';(', ':/', '=/', '=(', '(n)', 'D:'],
        'surprise':  [':o'],
        'love': ['<3'],
        'kiss': [':*'],
        'annoyed': ['-_-', '-__-'],
        'disappointed': ['.__.', '._.', ':|', ':\\', ':/'],
        'laugh': ['XD'],
        'thrill': [";')", ":')"]
    }

    for meaning in emoji_dictionary.keys():
        for (i, emoji) in enumerate(emoji_dictionary[meaning]):
            emoji = emoji.lower()
            spaced_emoji = ' '.join(list(emoji))
            text = text.replace(emoji, ' {} '.format(meaning))
            text = text.replace(spaced_emoji, ' {} '.format(meaning))

    # keep other emojis
    other_emojis = [':-','*)', '>.<']
    for (i, emoji) in enumerate(other_emojis):
        spaced_emoji = ' '.join(list(emoji))
        text = text.replace(emoji, ' <emoji{}> '.format(i))
        text = text.replace(spaced_emoji, ' <emoji{}> '.format(i))
    return text


# Normalize a piece of text
# Tweets are whitespace separated, have <user> and <url> already
def normalize_sentence(text: str) -> str:
    # lower case the text
    text = text.lower()

    # Remove whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove random numbers
    # Exclude: 4 you, 0 friends, < 3, me 2
    text = re.sub(r'\s[15-9]\s', ' ', text)
    # This needs to be done twice, find a better way
    text = re.sub(r'\s[0-9][0-9]+\s', '<num>', text)
    text = re.sub(r'\s[0-9][0-9]+\s', '<num>', text)

    # Remove weird non-printable characters
    text = ''.join([c for c in text if c in string.printable])

    # Specifics to the dataset
    text = re.sub(r'\( . . . \)', r' ', text)
    text = re.sub(r',', r'', text)

    # Reform Emoijis of the form (<char>) e.g. (y)
    text = re.sub(r'\(\s(?P<f1>\w)\s\)', r'(\1)', text)
    text = handle_emojis(text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove ., _, *, {, }, ', ", |, \, :, ~,`,^,-,=
    text = re.sub(r' \. ', r' ', text)
    text = re.sub(r'\\', r'', text)
    text = re.sub(r'\s_', r' ', text)
    text = re.sub(r'\*', r'', text)
    text = re.sub(r'\{', r'', text)
    text = re.sub(r'\}', r'', text)
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
    # Lematize, need to pass words individually
    text = ' '.join(nltk.WordNetLemmatizer().lemmatize(word) for word in text.split())

    return text
