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
