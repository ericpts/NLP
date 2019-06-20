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

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = False  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras
startTime = strftime('%d-%m-%Y_%H-%M-%S', localtime())


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
        save_weights_only=False,
        mode='max',
        period=1
    )

    # Setup tensorboard
    tensorboard = keras.callbacks.TensorBoard(
        log_dir='./logs' + '/' + model_name + startTime,
        histogram_freq=1 if model_name not in [
            "elmo",
            "elmomultilstm2",
            "elmomultilstm3",
            "elmomultilstm4",
            "elmomultilstm5"] else 0,
        update_freq=10000,
    )

    return [checkpoint, tensorboard]


def main(args: argparse.Namespace) -> None:
    os.system("mkdir -p models")
    os.system("mkdir -p checkpoints")
    os.system("mkdir -p logs")

    C['BATCH_SIZE'] = args.batch_size
    if args.model_name in ['elmomultilstm2', 'elmomultilstm3', 'elmomultilstm4', 'elmomultilstm5']:
        C['ELMO_SEQ'] = True

    text_input = args.model_name in ['elmo', 'elmomultilstm2', 'elmomultilstm3', 'elmomultilstm4', 'elmomultilstm5']
    model_path = os.path.join('models','{}.bin'.format(args.model_name))

    model = None
    if args.transfer != None:
        model = ModelBuilder.create_model(args.transfer)
    elif args.ensemble < 2:
        model = ModelBuilder.create_model(args.model_name)
    else:
        model = ModelBuilder.create_ensemble([args.model_name for i in range(args.ensemble)])

    if args.load != None:
        print("Loading model weights from: {}".format(args.load))
        model.load_weights(filepath=args.load)
        print("Model loaded from disk!")

    if args.transfer != None:
        model = ModelBuilder.create_model(args.model_name, [model])

    # Prints summary of the model
    model.summary()
    model.compile(
        loss='binary_crossentropy',
        metrics=['accuracy'],
        optimizer=Adam(lr=.001, decay=.0),
    )

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
        model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=get_callbacks(args.model_name))

        model.save(model_path)
        print("Model {} saved!".format(model_path))
    elif args.load is None:
        print("Loading previously trained .bin model from models/")
        print("You can specify a checkpoint to load from with --load")
        model = keras.models.load_model(model_path, custom_objects={'ElmoEmbeddingLayer': ElmoEmbedding.layer})
        print('Model loaded from disk.')

    # Predict using the test data
    X_test, _ = load_data(train=False, as_text=text_input)
    y_pred = model.predict(X_test)
    y_pred_debug = y_pred
    y_pred = [1 if pred > 0.5 else -1 for pred in y_pred]

    # Save predictions
    df = pd.DataFrame(y_pred, columns=['Prediction'], index=range(1, len(y_pred) + 1))
    df.index.name = 'Id'
    df.to_csv(PREDICTION_FILE)

    # Save predictions debug file
    df = pd.DataFrame(y_pred_debug, columns=['Prediction'], index=range(1, len(y_pred_debug) + 1))
    df.index.name = 'Id'
    df.to_csv(PREDICTION_DEBUG_FILE)


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
        "--transfer",
        type=str,
        help="Use for the loaded model for transfer learning.")
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
