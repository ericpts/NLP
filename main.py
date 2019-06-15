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

from constants import *
from util import *
from models import *


def get_callbacks(model_name: str) -> Callback:
    checkpoint_id = ''.join(random.choice(string.ascii_letters + string.digits)
        for i in range(5))

    checkpoint_name = "{}-{}-{}-{}.hdf5".format(
        model_name,
        checkpoint_id,
        "{epoch:02d}",
        "{val_acc:.2f}",
    )
    checkpoint_path = os.path.join('checkpoints', checkpoint_name)

    checkpoint = keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor='val_acc',
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode='max',
    )
    earlystop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        mode='min',
        verbose=1,
        patience=2,
    )

    # Setup tensorboard
    tensorboard = keras.callbacks.TensorBoard(
        log_dir='./logs',
        histogram_freq=1,
        update_freq=10000,
    )

    return [checkpoint, earlystop, tensorboard]


def main(args: argparse.Namespace) -> None:
    os.system("mkdir -p models")
    os.system("mkdir -p checkpoints")
    os.system("mkdir -p logs")

    model_path = os.path.join('models','{}.bin'.format(args.model_name))

    # Create model
    model = ModelBuilder.create_model(args.model_name)

    if args.load != None:
        print("Loading model weights from: {}".format(args.load))
        model.load_weights(filepath=args.load)
        print("Model loaded from disk!")

    # Prints summary of the model
    model.summary()
    model.compile(
        loss='binary_crossentropy',
        metrics=['accuracy'],
        optimizer=Adam(lr=.001, decay=.0),
    )

    if not args.eval:
        # Load and split data
        X, y = load_data(train=True, as_text=args.as_text)
        X_train, X_val, y_train, y_val = \
            train_test_split(X, y, test_size=TRAIN_TEST_SPLIT_PERCENTAGE)
        assert X_train.shape[0] == y_train.shape[0]
        assert X_val.shape[0] == y_val.shape[0]
        print('Train data: {}, Validation data: {}'.format(
            X_train.shape[0],
            X_val.shape[0],
        ))

        model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=get_callbacks(args.model_name))

        os.system("mkdir -p models")
        model.save(model_path)
        print("Model {} saved!".format(model_path))
    elif args.load is None:
        print("Loading previously trained .bin model from models/")
        print("You can specify a checkpoint to load from with --load")
        model = keras.models.load_model(model_path)
        print('Model loaded from disk.')

    # Predict using the test data
    X_test, _ = load_data(train=False, as_text=args.as_text)
    y_pred = model.predict(X_test).argmax(axis=-1)
    y_pred = [-1 if pred == 0 else pred for pred in y_pred]
    df = pd.DataFrame(y_pred, columns=['Prediction'], index=range(1, len(y_pred) + 1))
    df.index.name = 'Id'
    df.to_csv(PREDICTION_FILE)


if __name__ == '__main__':
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
        "--as-text",
        action='store_true',
        help="Use raw text for training."
    )
    parser.add_argument(
        'model_name',
        type=str,
        help="Model name (e.g. cnnlstm)"
    )

    main(parser.parse_args())
