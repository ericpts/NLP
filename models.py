import keras
import pickle
import random
import keras.layers as layers

from typing import List
from constants import *
from util import get_id

from keras.models import Model
from embeddings import ElmoEmbedding
from embeddings import Word2VecEmbedding
from embeddings import DefaultEmbedding


def _name_model(name: str) -> str:
    return "{}-{}".format(name, get_id())


class ElmoModels:
    @staticmethod
    def elmo() -> Model:
        # acc(train/valid/test): 0.87/0.84/0.84 | 10 epochs, commit b3ec | Adam lr 0.001
        inputs = layers.Input(shape=(1, ), dtype="string")
        X = inputs

        X = ElmoEmbedding.layer(elmo_type='default')(X)
        X = layers.Dense(512, activation='relu')(X)
        X = layers.Dropout(0.3)(X)
        X = layers.Dense(1, activation='sigmoid')(X)

        return Model(
            inputs=inputs,
            outputs=X,
            name=_name_model('elmo'))
        return model

    @staticmethod
    def elmobirnn() -> Model:
        inputs = layers.Input(shape=(MAX_SEQUENCE_LENGTH, ), dtype="string")

        X = inputs
        X = ElmoEmbedding.layer(elmo_type='elmo')(X)
        X = layers.normalization.BatchNormalization()(X)
        X = layers.Bidirectional(layers.GRU(
            units=128, dropout=.2, recurrent_dropout=.2, return_sequences=True))(X)
        X = layers.Bidirectional(layers.GRU(
            units=128, dropout=.2, recurrent_dropout=.2))(X)
        X = layers.Dropout(.5)(X)
        X = layers.Dense(64, activation='relu')(X)
        X = layers.Dense(1, activation='sigmoid')(X)

        return Model(
            inputs=inputs,
            outputs=X,
            name=_name_model('birnn'))
    
    @staticmethod
    def elmomultilstm() -> Model:
        inputs = layers.Input(shape=(MAX_SEQUENCE_LENGTH, ), dtype="string")

        X = inputs
        X = ElmoEmbedding.layer(elmo_type='elmo')(X)
        X = layers.LSTM(units=2048, return_sequences=True)(X)
        X = layers.LSTM(units=1024)(X)
        X = layers.Dropout(0.5)(X)
        X = layers.Dense(128, activation='relu')(X)
        X = layers.Dense(1, activation='sigmoid')(X)

        return Model(
            inputs=inputs,
            outputs=X,
            name=_name_model('elmomultilstm'))


class BaseModels:
    @staticmethod
    def birnn() -> Model:
        # acc(train/valid/test): 0.86/0.855/0.862 | 5 epochs, commit b3ec | Adam lr 0.0001
        inputs = layers.Input(shape=(MAX_SEQUENCE_LENGTH, ))

        X = inputs
        X = DefaultEmbedding.layer()(X)
        X = layers.normalization.BatchNormalization()(X)
        X = layers.Bidirectional(layers.GRU(
            units=128, dropout=.2, recurrent_dropout=.2, return_sequences=True))(X)
        X = layers.Bidirectional(layers.GRU(
            units=128, dropout=.2, recurrent_dropout=.2))(X)
        X = layers.Dropout(.5)(X)
        X = layers.Dense(64, activation='relu')(X)
        X = layers.Dense(1, activation='sigmoid')(X)

        return Model(
            inputs=inputs,
            outputs=X,
            name=_name_model('birnn'))

    @staticmethod
    def simple_rnn() -> Model:
        # acc(train/valid/test): 0.85/0.84/0.850 | 3 epochs, commit 4536 | Adam lr 0.001
        inputs = layers.Input(shape=(MAX_SEQUENCE_LENGTH, ))

        X = inputs
        X = DefaultEmbedding.layer()(X)
        X = layers.Dropout(.25)(X)
        X = layers.LSTM(units=50)(X)
        X = layers.Dropout(.25)(X)
        X = layers.Dense(1, activation='sigmoid')(X)

        return Model(
            inputs=inputs,
            outputs=X,
            name=_name_model('simple-rnn'))

    @staticmethod
    def cnn1layer() -> Model:
        # acc(train/valid/test): 0.85/0.84/0.853 | 5 epochs, commit 4536 | Adam lr 0.001
        inputs = keras.Input(shape=(MAX_SEQUENCE_LENGTH, ))

        X = inputs
        X = DefaultEmbedding.layer()(X)
        X = layers.Conv1D(64, 5, strides=1, padding='valid', activation='relu')(X)
        X = layers.MaxPooling1D(pool_size=2)(X)
        X = layers.Dropout(.5)(X)
        X = layers.Flatten()(X)
        X = layers.Dense(128, activation='relu')(X)
        X = layers.Dense(1, activation='sigmoid')(X)

        return Model(
            inputs=inputs,
            outputs=X,
            name=_name_model('cnn1layer'))

    @staticmethod
    def cnn_multiple_kernels() -> Model:
        # acc(train/valid/test): 0.84/0.84/0.855 | 2 epochs, commit a57e | Adam lr 0.001
        inputs = keras.Input(shape=(MAX_SEQUENCE_LENGTH, ))

        X = inputs
        X = DefaultEmbedding.layer()(X)

        # compute different local features; use 128 filters since it was underfitting
        bigram_branch = layers.Conv1D(filters=128,
            kernel_size=2, padding='valid', activation='relu', strides=1)(X)
        bigram_branch = layers.GlobalMaxPooling1D()(bigram_branch)

        trigram_branch = layers.Conv1D(filters=128,
            kernel_size=3, padding='valid', activation='relu', strides=1)(X)
        trigram_branch = layers.GlobalMaxPooling1D()(trigram_branch)

        fourgram_branch = layers.Conv1D(filters=128,
            kernel_size=4, padding='valid', activation='relu', strides=1)(X)
        fourgram_branch = layers.GlobalMaxPooling1D()(fourgram_branch)

        merged = layers.concatenate(
            [bigram_branch, trigram_branch, fourgram_branch], axis=1)

        # avoid filter codependency
        X = layers.Dropout(.5)(merged)
        # combine local features
        X = layers.Dense(128, activation='relu')(X)
        X = layers.Dense(1, activation='sigmoid')(X)

        return Model(
            inputs=inputs,
            outputs=X,
            name=_name_model('cnn-multiple-kernels'))


class TransferModels:
    '''
    Models that use trasnfer learning from previously trained models.
    '''

    @staticmethod
    def transfer_kernels(models: List[Model] = None) -> Model:
        # acc(train/valid/test): 0.87/0.87/0.863 | 2 epochs, commit ad1e | Adam lr 0.001
        # Pre: models[0] is birnn
        birnn = BaseModels.birnn() if models is None else models[0]
        inputs = layers.Input(shape=(MAX_SEQUENCE_LENGTH, ))

        X = inputs
        for layer in birnn.layers[:-4]:
            X = layer(X)
            layer.trainable = False

        Y = layers.Embedding(
            input_dim=MAX_WORDS,
            output_dim=256,
            input_length=MAX_SEQUENCE_LENGTH,
        )(inputs)
        X = layers.concatenate([X, Y], axis=1)

        bigram_branch = layers.Conv1D(filters=128,
            kernel_size=2, padding='valid', activation='relu', strides=1)(X)
        bigram_branch = layers.GlobalMaxPooling1D()(bigram_branch)
        trigram_branch = layers.Conv1D(filters=128,
            kernel_size=3, padding='valid', activation='relu', strides=1)(X)
        trigram_branch = layers.GlobalMaxPooling1D()(trigram_branch)
        fourgram_branch = layers.Conv1D(filters=128,
            kernel_size=4, padding='valid', activation='relu', strides=1)(X)
        fourgram_branch = layers.GlobalMaxPooling1D()(fourgram_branch)

        merged = layers.concatenate(
            [bigram_branch, trigram_branch, fourgram_branch], axis=1)

        X = layers.Dropout(.5)(merged)
        X = layers.Dense(128, activation='relu')(X)
        X = layers.Dense(1, activation='sigmoid')(X)

        return Model(
            inputs=inputs,
            outputs=X,
            name=_name_model('transfer-kernels'))

    @staticmethod
    def transfer_deeprnn(models: List[Model] = None) -> Model:
        # acc(train/valid/test): 0.87/0.88/0.862 | 2 epochs, commit ad1e | Adam lr 0.001
        # Pre: models[0] is birnn
        birnn = BaseModels.birnn() if models is None else models[0]

        for i in range(4):
            birnn.layers.pop()

        X = birnn.layers[-1].output
        for layer in birnn.layers:
            layer.trainable = False

        X = layers.Dropout(.5)(X)
        X = layers.Bidirectional(layers.GRU(
            units=128, dropout=.2, recurrent_dropout=.2, return_sequences=True))(X)
        X = layers.Bidirectional(layers.GRU(
            units=128, dropout=.2, recurrent_dropout=.2))(X)
        X = layers.Dropout(.5)(X)
        X = layers.Dense(64, activation='relu')(X)
        X = layers.Dense(1, activation='sigmoid')(X)

        return Model(
            inputs=birnn.inputs,
            outputs=X,
            name=_name_model('transfer-deeprnn'))


class Models(BaseModels, TransferModels, ElmoModels):
    pass


class ModelBuilder:
    '''
    Class used to create models and ensembles.
    '''

    models = {
        'simple-rnn': Models.simple_rnn,
        'cnn1layer': Models.cnn1layer,
        'cnn-multiple-kernels': Models.cnn_multiple_kernels,
        'birnn': Models.birnn,

        'transfer-deeprnn': Models.transfer_deeprnn,
        'transfer-kernels': Models.transfer_kernels,

        'elmo': Models.elmo,
        'elmobirnn': Models.elmobirnn,
        'elmomultilstm': Models.elmomultilstm,
    }

    @staticmethod
    def create_model(name: str, models: List[object] = None) -> Model:
        if models is None:
            return ModelBuilder.models[name]()
        else:
            return ModelBuilder.models[name](models)

    @staticmethod
    def get_model_input(model_name):
        if model_name in ['elmo', 'elmomultilstm']:
            return layers.Input(shape=(1, ), dtype="string")
        else:
            return keras.Input(shape=(MAX_SEQUENCE_LENGTH, ))

    @staticmethod
    def create_ensemble(names: List[str]) -> Model:
        '''
        Creates an ensemble of multiple models.
            Args:
            names            List of names of models to be added.
        '''
        inputs = ModelBuilder.get_model_input(names[0])
        models = [(name, ModelBuilder.create_model(name)) for name in names]
        outputs = [model(inputs) for name, model in models]

        avg_output = layers.average(outputs)
        return Model(
            inputs=inputs,
            outputs=avg_output,
            name='ensemble')
