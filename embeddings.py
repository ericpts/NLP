import gensim
from gensim import models
import tensorflow as tf
import numpy as np
from pathlib import Path
from nltk import word_tokenize
from constants import *
from util import *

class Word2Vec:

    @staticmethod
    def load_embedding():

        model = gensim.models.Word2Vec.load("word2vecTrainTest.model")
        print('man:', model['man'])
        print('woman:', model['woman'])
        print('king:', model['king'])
        print('queen:',model['queen'])
        print('res:', model['king'] - model['man'] + model['woman'])

    @staticmethod
    def map_extra_symbols(text):
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
    def learn_embeddings(files):
        sentences = []
        for f in files:
            for line in f:
                tokens = word_tokenize(Word2Vec.map_extra_symbols(normalize_sentence(line)))
                sentences.append([w for w in tokens if w.isalpha()])

        model = gensim.models.Word2Vec(sentences=sentences, size=EMBEDDING_DIM, window=5, workers=4, min_count=1)
        model.save("word2vecTrainTest.model")
        return sentences

if __name__ == '__main__':
    X_pos = Path(POSITIVE_TRAIN_DATA_FILE).read_text().split('\n')[:-1] # last one is empty
    X_neg = Path(NEGATIVE_TRAIN_DATA_FILE).read_text().split('\n')[:-1]
    X_test = Path(TEST_DATA_FILE).read_text().split('\n')[:-1] # Remove the index
    s = Word2Vec.learn_embeddings([X_pos, X_neg, X_test])
    print('LOADING EMBEDDING')
    Word2Vec.load_embedding()
