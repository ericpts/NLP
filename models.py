import tensorflow.keras as keras
import pickle
import numpy as np

from constants import *
from gensim.models import Word2Vec
from typing import List


class ModelBuilder():
    embedding_matrix = None
    word_index = None
    models = {}


    @staticmethod
    def initialize():
        ModelBuilder.models = {
            'simple-rnn' : ModelBuilder.simple_rnn,
        }
        try:
            with open('word_index.pkl', 'rb') as f:
                ModelBuilder.word_index = pickle.load(f)
        except:
            print('Pretrained embeddings not present')


    @staticmethod
    def create_model(name: str, usePretrainedEmbeddings: bool = True) -> keras.models.Model:
        if name not in ModelBuilder.models:
            print("Model {} not defined in the model_builder!".format(name))
            exit(1)

        ModelBuilder.usePretrainedEmbeddings = usePretrainedEmbeddings

        if usePretrainedEmbeddings:
            w2vmodel = Word2Vec.load("word2vecTrainTest.model")
            embeddings_index = w2vmodel.wv
            num_words = len(ModelBuilder.word_index) + 1

            # Map each word to an embedding, initially all of which are zeros
            ModelBuilder.embedding_matrix = np.zeros((num_words, EMBEDDING_DIM)) 
            for word, idx in ModelBuilder.word_index.items():
                if word in embeddings_index.vocab:
                    # Words not in the embedding index are all 0
                    ModelBuilder.embedding_matrix[idx] = embeddings_index[word]

        return ModelBuilder.models[name]()


    @staticmethod
    def create_ensemble(names: List[str], usePretrainedEmbeddings=True) -> keras.models.Model:
        '''
        Expect names of models in the ensemble
        '''
        models = [ModelBuilder.create_model(name, usePretrainedEmbeddings) for name in names]
        return ModelBuilder.ensemble_model(models)


    @staticmethod
    def get_embedding_layer(inputs):
        if not ModelBuilder.usePretrainedEmbeddings:
            return keras.layers.Embedding(MAX_WORDS, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH)(inputs)
        else:
            return keras.layers.Embedding(len(ModelBuilder.word_index) + 1,
                                       EMBEDDING_DIM,
                                       weights=[ModelBuilder.embedding_matrix],
                                       input_length=MAX_SEQUENCE_LENGTH,
                                       trainable=True)(inputs)


    @staticmethod
    def simple_rnn() -> keras.models.Model:
        inputs = keras.Input(shape=(MAX_SEQUENCE_LENGTH, ))

        X = inputs
        X = ModelBuilder.get_embedding_layer(X)
        X = keras.layers.Dropout(.25)(X)
        X = keras.layers.LSTM(
            units=MAX_SEQUENCE_LENGTH,
            input_shape=(EMBEDDING_DIM, 1))(X) # [TODO]: do I need recurrent dropout?
        X = keras.layers.Dropout(.25)(X)
        X = keras.layers.Dense(2, activation='softmax')(X)
        
        model = keras.models.Model(inputs=inputs, outputs=X, name='simple-rnn')
        return model


    @staticmethod
    def ensemble_model(*models) -> keras.models.Model:
        model_input = model_input = keras.Input(shape=(MAX_SEQUENCE_LENGTH, ))
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
