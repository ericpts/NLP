import tensorflow.keras as keras

from constants import *


def cnnlstm() -> keras.models.Model:
    inputs = keras.Input(shape=(MAX_SEQUENCE_LENGTH, ))

    X = inputs
    X = keras.layers.Embedding(MAX_WORDS, 128, input_length=MAX_SEQUENCE_LENGTH)(X)
    X = keras.layers.Dropout(1 / 4)(X)
    X = keras.layers.Conv1D(64, 5, strides=1, padding='same', activation='relu')(X)
    X = keras.layers.LSTM(64)(X)
    X = keras.layers.Dense(2, activation='softmax')(X)

    model = keras.models.Model(inputs=inputs, outputs=X, name='cnnlstm')
    return model


def cnn2layers() -> keras.models.Model:
    inputs = keras.Input(shape=(MAX_SEQUENCE_LENGTH, ))

    X = inputs
    X = keras.layers.Embedding(MAX_WORDS, 128, input_length=MAX_SEQUENCE_LENGTH)(X)
    X = keras.layers.Dropout(0.10)(X)
    X = keras.layers.Conv1D(64, 5, strides=1, padding='valid', activation='relu')(X)
    X = keras.layers.MaxPooling1D(pool_size=2)(X)
    X = keras.layers.Conv1D(128, 5, strides=1, padding='valid', activation='relu')(X)
    X = keras.layers.MaxPooling1D(pool_size=2)(X)
    X = keras.layers.Flatten()(X)
    X = keras.layers.Dense(64, activation='relu')(X)
    X = keras.layers.Dense(2, activation='softmax')(X)

    model = keras.models.Model(inputs=inputs, outputs=X, name='cnn2layers')
    return model

def cnn1layer() -> keras.models.Model:
    inputs = keras.Input(shape=(MAX_SEQUENCE_LENGTH, ))

    X = inputs
    X = keras.layers.Embedding(MAX_WORDS, 128, input_length=MAX_SEQUENCE_LENGTH)(X)
    X = keras.layers.Conv1D(64, 5, strides=1, padding='valid', activation='relu')(X)
    X = keras.layers.MaxPooling1D(pool_size=2)(X)
    X = keras.layers.Dropout(1 / 3)(X)
    X = keras.layers.Flatten()(X)
    X = keras.layers.Dense(128, activation='relu')(X)
    X = keras.layers.Dense(2, activation='softmax')(X)

    model = keras.models.Model(inputs=inputs, outputs=X, name='cnn1layer')
    return model

def multilstm() -> keras.models.Model:
    inputs = keras.Input(shape=(MAX_SEQUENCE_LENGTH, ))

    X = inputs
    X = keras.layers.Embedding(MAX_WORDS, 128, input_length=MAX_SEQUENCE_LENGTH)(X)
    X = keras.layers.LSTM(units=2048, return_sequences=True)(X)
    X = keras.layers.LSTM(units=1024)(X)
    X = keras.layers.Dropout(1 / 2)(X)
    X = keras.layers.Dense(128, activation='relu')(X)
    X = keras.layers.Dense(2, activation='softmax')(X)

    model = keras.models.Model(inputs=inputs, outputs=X, name='multilstm')
    return model

def cnnlstm2() -> keras.models.Model:
    inputs = keras.Input(shape=(MAX_SEQUENCE_LENGTH, ))

    X = inputs
    X = keras.layers.Embedding(MAX_WORDS, 128, input_length=MAX_SEQUENCE_LENGTH)(X)
    X = keras.layers.Conv1D(filters=64, kernel_size=5, activation='relu', padding='causal')(X)
    X = keras.layers.MaxPooling1D(pool_size=2)(X)
    X = keras.layers.Dropout(1 / 3)(X)
    X = keras.layers.LSTM(units=128)(X)
    X = keras.layers.Flatten()(X)
    X = keras.layers.Dense(64, activation='relu')(X)
    X = keras.layers.Dense(2, activation='softmax')(X)

    model = keras.models.Model(inputs=inputs, outputs=X, name='cnnlstm2')
    return model


def ensemble_models(*models):
    model_input = model_input = keras.Input(shape=(MAX_SEQUENCE_LENGTH, ))
    # Collect outputs of models
    outputs = [model(model_input) for model in models]
    # Average outputs
    avg_output = keras.layers.average(outputs)
    # Build model from same input and avg output
    modelEns = keras.models.Model(inputs=model_input, outputs=avg_output, name='ensemble')
    return modelEns


models = {
    'cnnlstm' : cnnlstm(),
    'cnn2layers' : ensemble_models(cnnlstm(), cnn2layers()),
}
