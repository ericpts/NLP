import keras
import tensorflow as tf
import pickle
import numpy as np

from constants import *
from gensim.models import Word2Vec
from typing import List

from keras.layers import Dense, Input, PReLU, Dropout
from embeddings import ElmoEmbedding, Word2Vec, DefaultEmbedding


class Models:
    @staticmethod
    def elmo() -> keras.models.Model:
        inputs = Input(shape=(1, ), name='input', dtype=tf.string)
        X = inputs

        X = ElmoEmbedding.layer()(X)
        X = Dense(512, activation='relu')(X)
        X = Dropout(0.3)(X)
        X = Dense(1, activation='sigmoid')(X)

        model = keras.models.Model(inputs=inputs, outputs=X, name='elmo')
        return model

    @staticmethod
    def simple_rnn() -> keras.models.Model:
        inputs = keras.Input(shape=(MAX_SEQUENCE_LENGTH, ))

        X = inputs
        X = DefaultEmbedding.layer()(X)
        X = keras.layers.Dropout(.25)(X)
        X = keras.layers.LSTM(
            units=MAX_SEQUENCE_LENGTH)(X) # [TODO]: do I need recurrent dropout?
        X = keras.layers.Dropout(.25)(X)
        X = keras.layers.Dense(2, activation='softmax')(X)

        model = keras.models.Model(inputs=inputs, outputs=X, name='simple-rnn')
        return model


class ModelBuilder:
    models = {
        'simple-rnn' : Models.simple_rnn,
        'elmo' : Models.elmo,
    }

    @staticmethod
    def create_model(name: str) -> keras.models.Model:
        return ModelBuilder.models[name]()


    @staticmethod
    def create_ensemble(names: List[str]) -> keras.models.Model:
        '''
        Expect names of models in the ensemble
        '''
        models = [ModelBuilder.create_model(name) for name in names]
        return ModelBuilder.ensemble_model(models)

    @staticmethod
    def ensemble_model(*models) -> keras.models.Model:
        model_input = keras.Input(shape=(MAX_SEQUENCE_LENGTH, ))
        # Collect outputs of models
        outputs = [model(model_input) for model in models]
        # Average outputs
        avg_output = keras.layers.average(outputs)
        # Build model from same input and avg output
        model_ensamble = keras.models.Model(
            inputs=model_input,
            outputs=avg_output,
            name='ensemble')
        return model_ensamble
