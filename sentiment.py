#!/usr/bin/env python3
###########################
# IMPORTS                 #
###########################
import argparse
import pandas as pd
from pathlib import Path
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split
from util import *

###########################
# MODEL                   #
###########################
def twitter_model():
    inputs = keras.Input(shape=(MAX_SEQUENCE_LENGTH, ))

    X = inputs
    X = keras.layers.Embedding(MAX_WORDS, 128, input_length=MAX_SEQUENCE_LENGTH)(X)
    X = keras.layers.Dropout(1 / 4)(X)
    X = keras.layers.Conv1D(64, 5, strides=1, padding='valid', activation='relu')(X)
    X = keras.layers.LSTM(64)(X)
    X = keras.layers.Dense(2)(X)

    model = keras.models.Model(inputs=inputs, outputs=X, name='TwitterModel')

    model.summary()
    return model

###########################
# MAIN                    #
###########################
def main(notrain):
    X, y = load_data(train=True)
    # Split train and val data in a 90-10 split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)

    assert X_train.shape[0] == y_train.shape[0]
    assert X_val.shape[0] == y_val.shape[0]
    # We can further split the train data to train and validation data
    print('Train data: {}, Validation data: {}'.format(X_train.shape[0], X_val.shape[0]))

    if notrain:
        if Path('model.bin').exists():
            model = keras.models.load_model('model.bin')
            print('Loaded model from disk.')
        else:
            print("Model couldn't be loaded from disk!")
            exit(1)
    else:
        model = twitter_model()

    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, validation_data=(X_test, y_test))

    model.save('model.bin')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--notrain",  action='store_true', help="Specify this option to not train and use the latest saved model")
    notrain = parser.parse_args().notrain
    main(notrain)
