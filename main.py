#!/usr/bin/env python3
import os
import argparse
import pandas as pd

from pathlib import Path
import tensorflow.keras as keras
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import train_test_split

from constants import *
from util import *
from models import *


def main() -> None:
    global ARGS
    model_path = os.path.join('models','{}.bin'.format(ARGS.model_name))
    
    # Create model
    ModelBuilder.initialize()
    model = ModelBuilder.create_model(ARGS.model_name, ARGS.pretrained_embeddings)

    if ARGS.load != None:
        print("Loading model weights from: {}".format(ARGS.load))
        model.load_weights(filepath=ARGS.load)
        print("Model loaded from disk!")

    # Prints summary of the model
    model.summary()
    model.compile(
        loss='binary_crossentropy',
        metrics=['accuracy'],
        optimizer=SGD(lr=0.01, momentum=0.9, clipnorm=5.0))

    if not ARGS.eval:
        # Load and split data
        X, y = load_data(train=True)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=TRAIN_TEST_SPLIT_PERCENTAGE)
        assert X_train.shape[0] == y_train.shape[0]
        assert X_val.shape[0] == y_val.shape[0]
        print('Train data: {}, Validation data: {}'.format(X_train.shape[0], X_val.shape[0]))

        # Setup callbacks
        # Checkpoint
        filepath= str(ARGS.model_name) + "-{epoch:02d}-{val_acc:.2f}.hdf5"
        checkpoint = keras.callbacks.ModelCheckpoint(
                        os.path.join('.', filepath),
                        monitor='val_acc',
                        verbose=1,
                        save_best_only=True,
                        save_weights_only=True,
                        mode='max')
        earlystop = keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        mode='min',
                        verbose=1,
                        patience=2)
        tensorboard = keras.callbacks.TensorBoard(
                        log_dir='./logs',
                        histogram_freq=1)
        callbacks_list = [checkpoint, tensorboard]

        model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=ARGS.epochs,
            batch_size=ARGS.batch_size,
            callbacks=callbacks_list)

        os.system("mkdir -p models")
        model.save(model_path)
        print("Model {} saved!".format(model_path))
    else:
        print("Loading previously trained .bin model from models/")
        print("You can specify a checkpoint to load from with --load")
        model = keras.models.load_model(model_path)
        print('Model loaded from disk.')

    # Predict using the test data
    X_test, _ = load_data(train=False)

    y_pred = model.predict(X_test).argmax(axis=-1)
    # Negative class is denoted by -1
    y_pred = [-1 if pred == 0 else pred for pred in y_pred]
    df = pd.DataFrame(y_pred, columns=['Prediction'], index=range(1, len(y_pred) + 1))
    df.index.name = 'Id'
    df.to_csv(PREDICTION_FILE)


if __name__ == '__main__':
    global ARGS
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval",
        action='store_true',
        help="Specify this option to not train but only eval using the last saved model or a specific checkpoint with --load")
    parser.add_argument(
        "--pretrained_embeddings",
        action='store_true',
        help="Specify this option to use pretrained embeddings instead of training new ones")
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
        'model_name',
        type=str,
        help="Model name (e.g. cnnlstm)"
    )

    ARGS = parser.parse_args()
    main()
