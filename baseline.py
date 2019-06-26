import numpy as np
import pandas as pd

from util import *
from constants import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from gensim.models import Word2Vec


if __name__ == "__main__":
    X, y = load_data(train=True, as_text=False)
    X_train, X_val, y_train, y_val = \
        train_test_split(X, y, test_size=TRAIN_TEST_SPLIT_PERCENTAGE) 

    assert X_train.shape[0] == y_train.shape[0]
    assert X_val.shape[0] == y_val.shape[0]

    # Get Word2Vec
    word_index = load_object(TOKENIZER_PATH).word_index
    
    w2vmodel = Word2Vec.load("models/word2vecTrainTest.model")
    embeddings_index = w2vmodel.wv
    num_words = len(word_index) + 1

    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    for word, idx in word_index.items():
        if word in embeddings_index.vocab:
            embedding_matrix[idx] = embeddings_index[word]
    
    preproc = lambda X: np.array([np.mean(
        [embedding_matrix[w] for i, w in enumerate(ws)], axis=0)
            for ws in X])

    # Train
    sgd = SGDClassifier()
    batch_size = 32
    for i in range(0, len(X_train), batch_size):
        X_batch = X_train[i:min(len(X_train), i + batch_size)]
        y_batch = y_train[i:min(len(y_train), i + batch_size)]
        
        if (i // batch_size) % 1000 == 0:
            print("Processing batch #{}".format(i // batch_size))

        X_batch = preproc(X_batch)
        sgd.partial_fit(X_batch, y_batch, classes=[0, 1])
    
    # Compute validation
    X_val = preproc(X_val)
    val = sgd.score(X_val, y_val)
    print("Validation: {}".format(val))

    # Predict
    X_test, _ = load_data(train=False, as_text=False)
    X_test = preproc(X_test)

    y_pred = sgd.predict(X_test)
    y_pred = [1 if pred > 0.5 else -1 for pred in y_pred]

    # Save predictions
    df = pd.DataFrame(
        y_pred,
        columns=['Prediction'],
        index=range(1, len(y_pred) + 1))
    df.index.name = 'Id'
    df.to_csv("preds/regr.csv")

