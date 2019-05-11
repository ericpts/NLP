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

        return ModelBuilder.models[name]()
