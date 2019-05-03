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
def twitter_model1():
    inputs = keras.Input(shape=(MAX_SEQUENCE_LENGTH, ))

    X = inputs
    X = keras.layers.Embedding(MAX_WORDS, 256, input_length=MAX_SEQUENCE_LENGTH)(X)
    X = keras.layers.Dropout(1 / 4)(X)
    X = keras.layers.Conv1D(64, 5, strides=1, padding='same', activation='relu')(X)
    X = keras.layers.LSTM(64)(X)
    X = keras.layers.Dense(2)(X)

    model = keras.models.Model(inputs=inputs, outputs=X, name='TwitterModel1')
    model.summary()
    return model

def twitter_model2():
    inputs = keras.Input(shape=(MAX_SEQUENCE_LENGTH, ))

    X = inputs
    X = keras.layers.Embedding(MAX_WORDS, 256, input_length=MAX_SEQUENCE_LENGTH)(X)
    X = keras.layers.Dropout(1 / 4)(X)
    X = keras.layers.LSTM(64, return_sequences=True)(X)
    X = keras.layers.Conv1D(64, 5, strides=1, padding='same', activation='relu')(X)
    X = keras.layers.Reshape((100*64, ))(X)
    X = keras.layers.Dense(2)(X)

    model = keras.models.Model(inputs=inputs, outputs=X, name='TwitterModel2')
    model.summary()
    return model

def ensembleModel(models):
    model_input = model_input = keras.Input(shape=(MAX_SEQUENCE_LENGTH, ))
    # Collect outputs of models
    outputs = [model(model_input) for model in models]
    # Average outputs
    avg_output = keras.layers.average(outputs)
    # Build model from same input and avg output
    modelEns = keras.models.Model(inputs=model_input, outputs=avg_output, name='ensemble')
    modelEns.summary()
    return modelEns



###########################
# MAIN                    #
###########################
def main(notrain):
    if notrain:
        if Path('model.bin').exists():
            model = keras.models.load_model('model.bin')
            print('Loaded model from disk.')
        else:
            print("Model couldn't be loaded from disk!")
            exit(1)
    else:
        X, y = load_data(train=True)
        # Split train and val data in a 90-10 split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)

        assert X_train.shape[0] == y_train.shape[0]
        assert X_val.shape[0] == y_val.shape[0]
        print('Train data: {}, Validation data: {}'.format(X_train.shape[0], X_val.shape[0]))

        # Make ensemble of these models
        models = [twitter_model1(), twitter_model2()]
        model = ensembleModel(models)
        model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, validation_data=(X_val, y_val))
        model.save('model.bin')
        print("Model saved!")

    # Predict using the test data
    X_test = load_data(train=False)

    y_pred = model.predict(X_test).argmax(axis=-1)
    # Negative class is denoted by -1
    y_pred = [-1 if pred == 0 else pred for pred in y_pred]
    df = pd.DataFrame(y_pred, columns=['Prediction'], index=range(1, len(y_pred) + 1))
    df.index.name = 'Id'
    df.to_csv(PREDICTION_FILE)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--notrain",  action='store_true', help="Specify this option to not train and use the latest saved model")
    notrain = parser.parse_args().notrain
    main(notrain)
