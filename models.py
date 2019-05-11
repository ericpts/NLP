import tensorflow.keras as keras
from constants import *
from gensim.models import Word2Vec
import numpy as np

class ModelBuilder():
    embedding_matrix = None
    word_index = None
    models = {}

    @staticmethod
    def initialize():
        ModelBuilder.models = {
            'cnnlstm' : ModelBuilder.cnnlstm,
            'cnn2layers' : ModelBuilder.cnn2layers,
            'cnn1layer': ModelBuilder.cnn1layer,
            'multilstm': ModelBuilder.multilstm,
            'cnnlstm2': ModelBuilder.cnnlstm2,
            'lstmcnn': ModelBuilder.lstmcnn,
        }

    @staticmethod
    def create_model(name, usePretrainedEmbeddings=True):
        if name not in ModelBuilder.models:
            print("Model {} not defined in the model_builder!".format(name))
            exit(1)

        ModelBuilder.usePretrainedEmbeddings = usePretrainedEmbeddings

        if usePretrainedEmbeddings:
            w2vmodel = Word2Vec.load("word2vecTrainTest.model")
            embeddings_index = w2vmodel.wv
            num_words = len(ModelBuilder.word_index) + 1
            ModelBuilder.embedding_matrix = np.zeros((num_words, EMBEDDING_DIM)) # Map each word to an embedding, initially all of which are zeros
            for word, idx in ModelBuilder.word_index.items():
                if word in embeddings_index.vocab:
                    # Words not in the embedding index are all 0
                    ModelBuilder.embedding_matrix[idx] = embeddings_index[word]

        return ModelBuilder.models[name]()

    @staticmethod
    def getEmbeddingLayer(input):
        if not ModelBuilder.usePretrainedEmbeddings:
            return keras.layers.Embedding(MAX_WORDS, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH)(input)
        else:
            return keras.layers.Embedding(len(ModelBuilder.word_index) + 1,
                                       EMBEDDING_DIM,
                                       weights=[ModelBuilder.embedding_matrix],
                                       input_length=MAX_SEQUENCE_LENGTH,
                                       trainable=False)(input)

    @staticmethod
    def cnnlstm() -> keras.models.Model:
        inputs = keras.Input(shape=(MAX_SEQUENCE_LENGTH, ))

        X = inputs
        X = ModelBuilder.getEmbeddingLayer(X)
        X = keras.layers.Dropout(1 / 4)(X)
        X = keras.layers.Conv1D(64, 5, strides=1, padding='same', activation='relu')(X)
        X = keras.layers.LSTM(64)(X)
        X = keras.layers.Dense(2, activation='softmax')(X)

        model = keras.models.Model(inputs=inputs, outputs=X, name='cnnlstm')
        return model

    @staticmethod
    def cnn2layers(embedding_matrix=None) -> keras.models.Model:
        inputs = keras.Input(shape=(MAX_SEQUENCE_LENGTH, ))
        X = inputs
        X = ModelBuilder.getEmbeddingLayer(X)
        X = keras.layers.Dropout(0.10)(X)
        X = keras.layers.Conv1D(64, 5, strides=1, padding='valid', activation='relu')(X)
        X = keras.layers.MaxPooling1D(pool_size=2)(X)
        X = keras.layers.Conv1D(128, 5, strides=1, padding='valid', activation='relu')(X)
        X = keras.layers.MaxPooling1D(pool_size=2)(X)
        X = keras.layers.Dropout(1 / 2)(X)
        X = keras.layers.Flatten()(X)
        X = keras.layers.Dense(64, activation='relu')(X)
        X = keras.layers.Dense(2, activation='softmax')(X)

        model = keras.models.Model(inputs=inputs, outputs=X, name='cnn2layers')
        return model

    @staticmethod
    def cnn1layer() -> keras.models.Model:
        inputs = keras.Input(shape=(MAX_SEQUENCE_LENGTH, ))

        X = inputs
        X = ModelBuilder.getEmbeddingLayer(X)
        X = keras.layers.Conv1D(64, 5, strides=1, padding='valid', activation='relu')(X)
        X = keras.layers.MaxPooling1D(pool_size=2)(X)
        X = keras.layers.Dropout(1 / 2)(X)
        X = keras.layers.Flatten()(X)
        X = keras.layers.Dense(128, activation='relu')(X)
        X = keras.layers.Dense(2, activation='softmax')(X)

        model = keras.models.Model(inputs=inputs, outputs=X, name='cnn1layer')
        return model

    @staticmethod
    def multilstm() -> keras.models.Model:
        inputs = keras.Input(shape=(MAX_SEQUENCE_LENGTH, ))

        X = inputs
        X = ModelBuilder.getEmbeddingLayer(X)
        X = keras.layers.LSTM(units=2048, return_sequences=True)(X)
        X = keras.layers.LSTM(units=1024)(X)
        X = keras.layers.Dropout(1 / 2)(X)
        X = keras.layers.Dense(128, activation='relu')(X)
        X = keras.layers.Dense(2, activation='softmax')(X)

        model = keras.models.Model(inputs=inputs, outputs=X, name='multilstm')
        return model

    @staticmethod
    def cnnlstm2() -> keras.models.Model:
        inputs = keras.Input(shape=(MAX_SEQUENCE_LENGTH, ))

        X = inputs
        X = ModelBuilder.getEmbeddingLayer(X)
        X = keras.layers.Conv1D(filters=64, kernel_size=5, activation='relu', padding='causal')(X)
        X = keras.layers.MaxPooling1D(pool_size=2)(X)
        X = keras.layers.Dropout(1 / 3)(X)
        X = keras.layers.LSTM(units=128)(X)
        X = keras.layers.Flatten()(X)
        X = keras.layers.Dense(64, activation='relu')(X)
        X = keras.layers.Dense(2, activation='softmax')(X)

        model = keras.models.Model(inputs=inputs, outputs=X, name='cnnlstm2')
        return model

    @staticmethod
    def lstmcnn() -> keras.models.Model:
        inputs = keras.Input(shape=(MAX_SEQUENCE_LENGTH, ))

        X = inputs
        X = ModelBuilder.getEmbeddingLayer(X)
        X = keras.layers.LSTM(units=2048, return_sequences=True)(X)
        X = keras.layers.Dropout(0.5)(X)
        X = tf.transpose(X, [0,2,1])
        X = keras.layers.Conv1D(filters=128, kernel_size=16, activation='relu', padding='valid')(X)
        X = keras.layers.MaxPooling1D(pool_size=8)(X)
        X = keras.layers.Dropout(0.25)(X)
        X = keras.layers.Conv1D(filters=64, kernel_size=8, activation='relu', padding='valid')(X)
        X = keras.layers.MaxPooling1D(pool_size=4)(X)
        X = keras.layers.Dropout(0.15)(X)
        X = keras.layers.Flatten()(X)
        X = keras.layers.Dense(128, activation='relu')(X)
        X = keras.layers.Dense(2, activation='softmax')(X)

        model = keras.models.Model(inputs=inputs, outputs=X, name='lstmcnn')
        return model
