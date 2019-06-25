import os
import sys
os.environ['TF_KERAS'] = '1'

import argparse
import itertools
import pandas as pd
import keras_bert
import tensorflow as tf
import tensorflow.keras as keras
import codecs
import numpy as np
import util
import constants
import pickle
import time

from typing import Tuple
from pathlib import Path
from tensorflow.python.ops.math_ops import erf, sqrt
from sklearn.model_selection import train_test_split


def gelu(x):
    return 0.5 * x * (1.0 + erf(x / sqrt(2.0)))


keras_bert.bert.gelu = gelu


BATCH_SIZE = 64
EPOCHS = 100
STEPS_PER_EPOCH = 2000


def get_bert_model_dir() -> Path:
    '''
    Returns the path to the Bert pretrined model directory.
    '''
    small_model_dir = ".models/uncased_L-12_H-768_A-12"
    big_model_dir = ".models/uncased_L-24_H-1024_A-16"
    return Path(big_model_dir)


def get_bert_model() -> keras.models.Model:
    '''
    Returns the Bert model.
    '''
    config_path = get_bert_model_dir() / 'bert_config.json'
    checkpoint_path = get_bert_model_dir() / 'bert_model.ckpt'

    bert_model = keras_bert.load_trained_model_from_checkpoint(
        str(config_path),
        str(checkpoint_path),
        training=True,
        trainable=True,
        seq_len=constants.MAX_SEQUENCE_LENGTH
    )
    (indices, always_zero) = bert_model.inputs[:2]
    dense = bert_model.get_layer('NSP-Dense').output
    bert_model = keras.models.Model(
        inputs=(indices, always_zero),
        outputs=dense,
        name='bert_embeddings'
    )

    # Build the actual model.
    inputs = indices
    X = bert_model(
        [inputs, tf.zeros_like(inputs)]
    )
    X = keras.layers.Dense(1, activation='sigmoid')(X)

    return keras.models.Model(
        inputs=indices,
        outputs=X,
    )


def get_tokenizer() -> keras_bert.Tokenizer:
    '''
    Returns the Bert tokenizer.
    '''
    cache_f = Path('datasets/bert_tokenizer.bin')
    if cache_f.exists():
        print('Found cached tokenizer.')
        with cache_f.open('r+b') as f:
            tokenizer = pickle.load(f)
            return tokenizer

    print('Building tokenizer.')
    vocab_path = get_bert_model_dir() / 'vocab.txt'

    token_dict = {}
    with codecs.open(str(vocab_path), 'r', 'utf8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)

    at_unused = 0

    def next_unused() -> str:
        nonlocal at_unused
        ret = f'[unused{at_unused}]'
        at_unused += 1
        return ret

    def add_token_to_vocab(token: str):
        to_replace = next_unused()
        assert to_replace in token_dict
        token_dict[token] = token_dict[to_replace]
        del token_dict[to_replace]

    add_token_to_vocab('<user>')
    add_token_to_vocab('<url>')
    add_token_to_vocab('<num>')

    tokenizer = keras_bert.Tokenizer(token_dict)
    with cache_f.open('w+b') as f:
        pickle.dump(tokenizer, f)
    return tokenizer


def get_bert_data(train: bool) -> Tuple[np.array, np.array]:
    '''
    Returns data for training or evaluation.

    Args:
        train: Whether we want the training data.
    '''
    cache_f = Path(f'datasets/bert_{train}.npz')
    if cache_f.exists():
        data = np.load(str(cache_f))
        return data['X'], data['y']

    print('Computing bert data...')

    tokenizer = get_tokenizer()

    X, y = util.load_data(train=train, as_text=True)
    X = [tokenizer.encode(x, max_len=constants.MAX_SEQUENCE_LENGTH)[0] for x in X]
    X = np.array(X)

    np.savez(
        str(cache_f),
        X=X,
        y=y
    )
    return X, y


def predict(args: argparse.Namespace) -> None:
    """
    Predicts outputs given path to weights of the Bert model.
    """
    weights = Path(args.weights)
    assert weights.exists()

    model = get_bert_model()
    model.load_weights(str(weights))
    print('Loaded model.')
    X_test, _ = get_bert_data(train=False)

    y_pred = model.predict(X_test)
    y_pred = [1 if pred > 0.5 else -1 for pred in y_pred]
    df = pd.DataFrame(y_pred, columns=['Prediction'], index=range(1, len(y_pred) + 1))
    df.index.name = 'Id'
    df.to_csv('bert_out.csv')


def train(args: argparse.Namespace) -> None:
    """
    Tune the Bert model, given the pretrained one.
    """
    model = get_bert_model()
    model.summary()

    if args.weights:
        weights = Path(args.weights)
        assert weights.exists()
        model.load_weights(str(weights))
        print(f'Loaded weights from {weights}.')

    X, y = get_bert_data(True)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=constants.TRAIN_TEST_SPLIT_PERCENTAGE)

    D_train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    D_train = D_train.shuffle(y_train.shape[0])
    D_train = D_train.batch(BATCH_SIZE)
    D_train = D_train.repeat()

    D_val = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    D_val = D_val.batch(BATCH_SIZE)

    total_steps = STEPS_PER_EPOCH * EPOCHS
    warmup_steps = total_steps // 10
    decay_steps = total_steps - warmup_steps

    print(f'decay_steps = {decay_steps}, warmup_steps = {warmup_steps}', flush=True)

    lr = float(args.lr)
    print(f'Using a learning rate of {lr}')

    model.compile(
        optimizer=keras_bert.AdamWarmup(
            decay_steps=decay_steps,
            warmup_steps=warmup_steps,
            lr=lr,
            min_lr=1e-7,
            epsilon=1e-3,
        ),
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )

    filepath = Path('models') / str(int(time.time())) / "bert-{epoch:02d}-{val_accuracy:.4f}.hdf5"
    filepath.parent.mkdir(exist_ok=True, parents=True)

    checkpoint = keras.callbacks.ModelCheckpoint(
        str(filepath),
        monitor='val_accuracy',
        verbose=1,
        save_weights_only=True,
        mode='max',
        period=2,
    )

    callbacks_list = [
        checkpoint,
    ]

    model.fit(
        D_train,
        validation_data=D_val,
        epochs=EPOCHS,
        callbacks=callbacks_list,
        steps_per_epoch=STEPS_PER_EPOCH,
    )


def main(args: argparse.Namespace) -> None:
    if not get_bert_model_dir().exists():
        print("Bert model has not been downloaded.")
        sys.exit(1)

    if args.predict:
        if not args.weights or not Path(args.weights).exists():
            print("Bert weights not present.")
            sys.exit(1)
        predict(args)
    else:
        train(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--weights',
        type=str,
        help='Where to load the weights from.')
    parser.add_argument(
        '--prediction',
        action='store_true',
        help="Use the model for prediction.")
    parser.add_argument(
        '--lr',
        type=str,
        default='2e-5',
        help='Learning rate.')
    args = parser.parse_args()
    main(args)
