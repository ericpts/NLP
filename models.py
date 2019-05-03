import tensorflow.keras as keras

from constants import *


def twitter_model1() -> keras.models.Model:
    inputs = keras.Input(shape=(MAX_SEQUENCE_LENGTH, ))

    X = inputs
    X = keras.layers.Embedding(MAX_WORDS, 128, input_length=MAX_SEQUENCE_LENGTH)(X)
    X = keras.layers.Dropout(1 / 4)(X)
    X = keras.layers.Conv1D(64, 5, strides=1, padding='same', activation='relu')(X)
    X = keras.layers.LSTM(64)(X)
    X = keras.layers.Dense(2)(X)

    model = keras.models.Model(inputs=inputs, outputs=X, name='TwitterModel1')
    return model


def twitter_model2() -> keras.models.Model:
    inputs = keras.Input(shape=(MAX_SEQUENCE_LENGTH, ))

    X = inputs
    X = keras.layers.Embedding(MAX_WORDS, 128, input_length=MAX_SEQUENCE_LENGTH)(X)
    X = keras.layers.Dropout(1 / 4)(X)
    X = keras.layers.LSTM(64, return_sequences=True)(X)
    X = keras.layers.Conv1D(64, 5, strides=1, padding='same', activation='relu')(X)
    X = keras.layers.Reshape((MAX_SEQUENCE_LENGTH * 64, ))(X)
    X = keras.layers.Dense(2)(X)

    model = keras.models.Model(inputs=inputs, outputs=X, name='TwitterModel2')
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
    'twitter_model1' : twitter_model1(),
    'twitter_model2' : ensemble_models(twitter_model1(), twitter_model2()),
}
