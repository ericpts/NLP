import os
import sys
import argparse
import gensim

import tensorflow as tf
import tensorflow_hub as hub
import keras.backend as K
import keras.layers as layers
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

    @staticmethod
    def layer() -> Layer:
        raise NotImplementedError()


class Word2VecEmbedding(Embedding):
    @staticmethod
    def layer() -> Layer:
        word_index = load_object(TOKENIZER_PATH).word_index

        w2vmodel = gensim.models.Word2Vec.load("models/word2vecTrainTest.model")
        embeddings_index = w2vmodel.wv
        num_words = len(word_index) + 1

        # Map each word to an embedding, initially all of which are zeros
        embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
        for word, idx in word_index.items():
            if word in embeddings_index.vocab:
                # Words not in the embedding index are all 0
                embedding_matrix[idx] = embeddings_index[word]

        return layers.Embedding(
            len(word_index) + 1,
            EMBEDDING_DIM,
            weights=[embedding_matrix],
            input_length=MAX_SEQUENCE_LENGTH,
            trainable=True)

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
        model.save("models/word2vecTrainTest.model")

        return sentences


class ElmoEmbeddingLayer(Layer):
    def __init__(self,
        trainable=True,
        batch_size=32,
        elmo_type='default',
        **kwargs
    ):
        self.batch_size = batch_size
        self.dimensions = 1024
        self.trainable = trainable
        self.type = elmo_type
        super(ElmoEmbeddingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.elmo = hub.Module(
            'https://tfhub.dev/google/elmo/2',
            trainable=self.trainable,
            name="{}_module".format(self.name))

        self.trainable_weights += K.tf.trainable_variables(scope="^{}_module/.*".format(self.name))
        super(ElmoEmbeddingLayer, self).build(input_shape)

    def call(self, x, mask=None):
        return self.elmo(
            inputs={
                'tokens': K.cast(x, tf.string),
                'sequence_len': tf.constant(self.batch_size * [MAX_SEQUENCE_LENGTH])
            },
            as_dict=True,
            signature='tokens',
        )[self.type]

    def compute_mask(self, inputs, mask=None):
        return K.not_equal(inputs, '--PAD--')

    def compute_output_shape(self, input_shape):
        return (input_shape[0], MAX_SEQUENCE_LENGTH, self.dimensions)


class ElmoEmbedding(Embedding):
    @staticmethod
    def learn_embeddings(files : List[str]) -> List[str]:
        return []

    @staticmethod
    def layer(elmo_type: str= 'default', trainable: bool = True) -> Layer:
        return ElmoEmbeddingLayer(elmo_type=elmo_type, trainable=trainable)


class DefaultEmbedding(Embedding):
    @staticmethod
    def learn_embeddings(files : List[str]) -> List[str]:
        return []

    @staticmethod
    def layer() -> Layer:
        return layers.Embedding(
            input_dim=MAX_WORDS,
            output_dim=EMBEDDING_DIM,
            input_length=MAX_SEQUENCE_LENGTH,
        )


def main(args: argparse.Namespace) -> None:
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


if __name__ == '__main__':
    init()

    embeddings = {
        "word2vec" : Word2VecEmbedding,
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
        main(ns)
        sys.exit(0)
    parser.print_help()
