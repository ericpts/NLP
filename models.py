import keras
import tensorflow as tf
import pickle
import numpy as np

from constants import *
from typing import List

from embeddings import ElmoEmbedding, Word2Vec, DefaultEmbedding

class Models:
    @staticmethod
    def elmo() -> keras.models.Model:
        inputs = keras.layers.Input(shape=(1, ), dtype=tf.string)
        X = inputs

        X = ElmoEmbedding.layer()(X)
        X = keras.layers.Dense(512, activation='relu')(X)
        X = keras.layers.Dropout(0.3)(X)
        X = keras.layers.Dense(1, activation='sigmoid')(X)

        model = keras.models.Model(inputs=inputs, outputs=X, name='elmo')
        return model

    @staticmethod
    def simple_rnn() -> keras.models.Model:
        # acc(train/valid/test): 0.85/0.84/0.823 | 3 epochs, commit 4536 | Adam lr 0.001
        inputs = keras.layers.Input(shape=(MAX_SEQUENCE_LENGTH, ))

        X = inputs
        X = DefaultEmbedding.layer()(X)
        X = keras.layers.Dropout(.25)(X)
        X = keras.layers.LSTM(units=50)(X)
        X = keras.layers.Dropout(.25)(X)
        X = keras.layers.Dense(1, activation='sigmoid')(X)

        model = keras.models.Model(inputs=inputs, outputs=X, name='simple-rnn')
        return model

    @staticmethod
    def cnn1layer() -> keras.models.Model:
        # acc(train/valid/test): 0.85/0.84/0.823 | 5 epochs, commit 4536 | Adam lr 0.001
        inputs = keras.Input(shape=(MAX_SEQUENCE_LENGTH, ))

        X = inputs
        X = DefaultEmbedding.layer()(X)
        X = keras.layers.Conv1D(64, 5, strides=1, padding='valid', activation='relu')(X)
        X = keras.layers.MaxPooling1D(pool_size=2)(X)
        X = keras.layers.Dropout(.5)(X)
        X = keras.layers.Flatten()(X)
        X = keras.layers.Dense(128, activation='relu')(X)
        X = keras.layers.Dense(1, activation='sigmoid')(X)

        model = keras.models.Model(inputs=inputs, outputs=X, name='cnn1layer')
        return model

    @staticmethod
    def cnn_multiple_kernels() -> keras.models.Model:
        # acc(train/valid/test): 0.84/0.84/0.829 | 2 epochs, commit a57e | Adam lr 0.001
        inputs = keras.Input(shape=(MAX_SEQUENCE_LENGTH, ))

        X = inputs
        X = DefaultEmbedding.layer()(X)

        # compute different local features; use 128 filters since it was underfitting
        bigram_branch = keras.layers.Conv1D(filters=128,
            kernel_size=2, padding='valid', activation='relu', strides=1)(X)
        bigram_branch = keras.layers.GlobalMaxPooling1D()(bigram_branch)

        trigram_branch = keras.layers.Conv1D(filters=128,
            kernel_size=3, padding='valid', activation='relu', strides=1)(X)
        trigram_branch = keras.layers.GlobalMaxPooling1D()(trigram_branch)

        fourgram_branch = keras.layers.Conv1D(filters=128,
            kernel_size=4, padding='valid', activation='relu', strides=1)(X)
        fourgram_branch = keras.layers.GlobalMaxPooling1D()(fourgram_branch)

        merged = keras.layers.concatenate(
            [bigram_branch, trigram_branch, fourgram_branch], axis=1)

        # avoid filter codependency
        X = keras.layers.Dropout(.5)(merged)
        # combine local features
        X = keras.layers.Dense(128, activation='relu')(X)
        X = keras.layers.Dense(1, activation='sigmoid')(X)
        model = keras.models.Model(inputs=inputs, outputs=X, name='cnn-multiple-kernels')
        return model


class ModelBuilder:
    models = {
        'simple-rnn' : Models.simple_rnn,
        'cnn1layer' : Models.cnn1layer,
        'elmo' : Models.elmo,
        'cnn-multiple-kernels' : Models.cnn_multiple_kernels,
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
    def ensemble_model(*models: List[keras.models.Model]) -> keras.models.Model:
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
