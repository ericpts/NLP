import tensorflow.keras as keras
from gensim.models import Word2Vec
import pickle
import numpy as np
from constants import *

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
            'nlp': ModelBuilder.nlp,
        }
        with open('word_index.pkl', 'rb') as f:
            ModelBuilder.word_index = pickle.load(f)

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
    def create_ensemble(names, usePretrainedEmbeddings=True):
        '''
        Expect names of models in the ensemble
        '''
        models = [ModelBuilder.create_model(name, usePretrainedEmbeddings) for name in names]
        return ModelBuilder.ensemble_model(models)

    @staticmethod
    def getEmbeddingLayer(inputs):
        if not ModelBuilder.usePretrainedEmbeddings:
            return keras.layers.Embedding(MAX_WORDS, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH)(inputs)
        else:
            return keras.layers.Embedding(len(ModelBuilder.word_index) + 1,
                                       EMBEDDING_DIM,
                                       weights=[ModelBuilder.embedding_matrix],
                                       input_length=MAX_SEQUENCE_LENGTH,
                                       trainable=True)(inputs)

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
        X = keras.layers.Dropout(0.5)(X)
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

    @staticmethod
    def nlp() -> keras.models.Model:
        inputs = keras.Input(shape=(MAX_SEQUENCE_LENGTH, ))

        X = inputs
        X = ModelBuilder.getEmbeddingLayer(X)
        bigram_branch = keras.layers.Conv1D(filters=100, kernel_size=2, padding='valid', activation='relu', strides=1)(X)
        bigram_branch = keras.layers.GlobalMaxPooling1D()(bigram_branch)
        trigram_branch = keras.layers.Conv1D(filters=100, kernel_size=3, padding='valid', activation='relu', strides=1)(X)
        trigram_branch = keras.layers.GlobalMaxPooling1D()(trigram_branch)
        fourgram_branch = keras.layers.Conv1D(filters=100, kernel_size=4, padding='valid', activation='relu', strides=1)(X)
        fourgram_branch = keras.layers.GlobalMaxPooling1D()(fourgram_branch)
        merged = keras.layers.concatenate([bigram_branch, trigram_branch, fourgram_branch], axis=1)

        merged = keras.layers.Dense(256, activation='relu')(merged)
        merged = keras.layers.Dropout(0.2)(merged)
        merged = keras.layers.Dense(2, activation='softmax')(merged) # TODO: make 1
        model = keras.models.Model(inputs=inputs, outputs=merged, name='nlp')
        return model



    @staticmethod
    def ensemble_model(*models) -> keras.models.Model:
        model_input = model_input = keras.Input(shape=(MAX_SEQUENCE_LENGTH, ))
        # Collect outputs of models
        outputs = [model(model_input) for model in models]
        # Average outputs
        avg_output = keras.layers.average(outputs)
        # Build model from same input and avg output
        modelEns = keras.models.Model(inputs=model_input, outputs=avg_output, name='ensemble')
        return modelEns
