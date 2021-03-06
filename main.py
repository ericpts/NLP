#!/usr/bin/env python3
import os
import random
import string
import argparse
import pandas as pd
import keras

from keras.optimizers import SGD, Adam
from keras.callbacks import Callback
from sklearn.model_selection import train_test_split
from typing import List
from pathlib import Path
from time import strftime, localtime

from embeddings import ElmoEmbedding
from constants import *
from util import *
from models import *


def _get_callbacks(model_name: str, session_id: str) -> Callback:
    '''
    Returns the callbacks to append for training.
    '''
    checkpoint_name = "{}-{}-{}-{}.hdf5".format(
        model_name,
        session_id,
        "{epoch:02d}",
        "{val_acc:.2f}",
    )
    checkpoint_path = os.path.join('checkpoints', checkpoint_name)

    checkpoint = keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor='val_acc',
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode='max',
        period=1
    )

    # Setup tensorboard
    tensorboard = keras.callbacks.TensorBoard(
        log_dir="./logs",
        histogram_freq=1 if model_name not in [
            "elmo",
            "elmobirnn",
            "elmomultilstm"] else 0,
        update_freq=10000,
    )

    return [checkpoint, tensorboard]


def main(args: argparse.Namespace) -> None:
    text_input = args.model_name in ['elmo', 'elmobirnn', 'elmomultilstm']
    model_path = os.path.join('models','{}.bin'.format(args.model_name))
    session_id = get_id()

    model = None
    if args.transfer != None:
        # Get the model for tranfer learning
        model = ModelBuilder.create_model(args.transfer)
    elif args.ensemble < 2:
        # Create a single model
        model = ModelBuilder.create_model(args.model_name)
    else:
        # Create an ensemble of multiple similar models
        model = ModelBuilder.create_ensemble([
            args.model_name for i in range(args.ensemble)])

    # Load weights from disk
    if args.load != None:
        print("Loading model weights from: {}".format(args.load))
        model.load_weights(filepath=args.load)
        print("Model loaded from disk!")

    # Create the model if we reuse a model for transfer learning
    if args.transfer != None:
        model = ModelBuilder.create_model(args.model_name, [model])

    # Compile the model
    model.summary()
    model.compile(
        loss='binary_crossentropy',
        metrics=['accuracy'],
        optimizer=Adam(lr=.0001, decay=.0),
    )

    # We are running in usual model, rather than evaluation mode
    if not args.eval:
        # Load and split data
        X, y = load_data(train=True, as_text=text_input)
        X_train, X_val, y_train, y_val = \
            train_test_split(X, y, test_size=TRAIN_TEST_SPLIT_PERCENTAGE)

        assert X_train.shape[0] == y_train.shape[0]
        assert X_val.shape[0] == y_val.shape[0]
        print('Train data: {}, Validation data: {}'.format(
            X_train.shape[0],
            X_val.shape[0],

        ))

        # Train the model
        model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=_get_callbacks(args.model_name, session_id))

        # Save the final weights
        model.save(model_path)
        print("Model {} saved!".format(model_path))
    # In case we have saved the model from a previous run
    elif args.load is None:
        print("Loading previously trained .bin model from models/")
        print("You can specify a checkpoint to load from with --load")
        model = keras.models.load_model(model_path, custom_objects={
            'ElmoEmbeddingLayer': ElmoEmbedding.layer,
        })
        print('Model loaded from disk.')

    # Predict using the test data
    X_test, _    = load_data(train=False, as_text=text_input)
    y_pred       = model.predict(X_test)
    y_pred_debug = y_pred
    y_pred       = [1 if pred > 0.5 else -1 for pred in y_pred]

    # Save predictions
    df = pd.DataFrame(
        y_pred,
        columns=['Prediction'],
        index=range(1, len(y_pred) + 1))
    df.index.name = 'Id'
    df.to_csv("preds/{}-{}.csv".format(args.model_name, session_id))

    # Save predictions debug file
    df = pd.DataFrame(
        y_pred_debug,
        columns=['Prediction'],
        index=range(1, len(y_pred_debug) + 1))
    df.index.name = 'Id'
    df.to_csv("preds/{}-{}-debug.csv".format(args.model_name, session_id))


if __name__ == '__main__':
    init()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval",
        action='store_true',
        help="Specify this option to not train but only eval using the last saved model or a specific checkpoint with --load")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size to use during training.")
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of epochs to train for")
    parser.add_argument(
        "--load",
        type=str,
        help="Specify some checkpoint to load. Specify the .hdf5 file without .data or .index afterwards")
    parser.add_argument(
        "--transfer",
        type=str,
        help="Use the loaded model for transfer learning.")
    parser.add_argument(
        'model_name',
        type=str,
        help="Model name (e.g. cnnlstm)"
    )
    parser.add_argument(
        '--ensemble',
        type=int,
        default=1,
        help="Ensemble size to use"
    )
    main(parser.parse_args())
