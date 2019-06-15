import os
import sys
import argparse
import gensim
import tensorflow as tf
import tensorflow_hub as hub
import keras.backend as K
import numpy as np

from pathlib import Path
from gensim import models
from nltk import word_tokenize
from constants import *
from util import *

from keras.models import Model
from keras.engine.topology import Layer
from typing import List


class Embedding(object):
    @staticmethod
    def learn_embeddings(files : List[str]) -> List[str]:
        raise NotImplementedError()


class Word2Vec(Embedding):
    @staticmethod
    def debug_embedding() -> None:
        model = gensim.models.Word2Vec.load("models/word2vecTrainTest.model")
        print('man:', model['man'])
        print('woman:', model['woman'])
        print('king:', model['king'])
        print('queen:',model['queen'])
        print('res:', model['king'] - model['man'] + model['woman'])


    @staticmethod
    def __map_extra_symbols(text : str) -> str:
        emoji_dictionary = {
            'exclamation': [i * '!' for i in range(1, 11, 1)],
            'pause': [i * '.' for i in range(1,  11, 1)],
            'question': [i * '?' for i in range(1, 11, 1)],
        }
        for meaning in emoji_dictionary.keys():
            for (i, emoji) in enumerate(emoji_dictionary[meaning]):
                spaced_emoji = ' '.join(list(emoji))
                text = text.replace(emoji, ' {} '.format(meaning))
                text = text.replace(spaced_emoji, ' {} '.format(meaning))
        return text


    @staticmethod
    def learn_embeddings(files : List[str]) -> List[str]:
        sentences = []
        for f in files:
            for line in f:
                tokens = word_tokenize(
                    Word2Vec.__map_extra_symbols(normalize_sentence(line)))
                sentences.append([w for w in tokens if w.isalpha()])

        model = gensim.models.Word2Vec(
            sentences=sentences,
            size=EMBEDDING_DIM,
            window=5,
            workers=4,
            min_count=1)
        os.system("mkdir -p models")
        model.save("models/word2vecTrainTest.model")

        return sentences


class ElmoEmbeddingLayer(Layer):
    def __init__(self, trainable=True, **kwargs):
        self.dimensions = 1024
        self.trainable = trainable
        super(ElmoEmbeddingLayer, self).__init__(**kwargs)


    def build(self, input_shape):
        self.elmo = hub.Module(
            'https://tfhub.dev/google/elmo/2',
            trainable=self.trainable,
            name="{}_module".format(self.name))

        self.trainable_weights += K.tf.trainable_variables(scope="^{}_module/.*".format(self.name))
        super(ElmoEmbeddingLayer, self).build(input_shape)


    def call(self, x, mask=None):
        result = self.elmo(
            K.squeeze(K.cast(x, tf.string), axis=1),
            as_dict=True,
            signature='default',
        )['default']
        return result


    def compute_mask(self, inputs, mask=None):
        return K.not_equal(inputs, '--PAD--')


    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.dimensions)


def gen_model():
    inputs = Input(shape=(1, ), name='input', dtype=tf.string)
    X = inputs

    X = ElmoEmbeddingLayer()(X)

    X = Dense(512, activation='relu')(X)
    X = Dropout(0.3)(X)
    X = Dense(1, activation='sigmoid')(X)

    model = Model(
        inputs=inputs,
        outputs=X,
        name='elmo'
    )

    return model

if __name__ == '__main__':
    embeddings = {
        "word2vec" : Word2Vec,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train",
        type=str,
        help="Specify which embedding to train out of: {}".format(
            list(embeddings.keys())
        ),
    )

    ns = parser.parse_args()
    if ns.train:
        if ns.train not in embeddings:
            print("{} model not present. Possible embeddings: {}".format(
                ns.train,
                list(embeddings.keys()),
            ))
            sys.exit(1)

        model = embeddings[ns.train]

        X_pos = Path(POSITIVE_TRAIN_DATA_FILE).read_text().split('\n')[:-1] # last one is empty
        X_neg = Path(NEGATIVE_TRAIN_DATA_FILE).read_text().split('\n')[:-1]
        X_test = Path(TEST_DATA_FILE).read_text().split('\n')[:-1] # Remove the index
        s = model.learn_embeddings([X_pos, X_neg, X_test])

        print('LOADING EMBEDDING')
        Word2Vec.debug_embedding()
        sys.exit(0)

    parser.print_help()
    gen_model()
