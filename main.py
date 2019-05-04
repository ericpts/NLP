#!/usr/bin/env python3
import os
import argparse
import pandas as pd

from pathlib import Path
import tensorflow.keras as keras
from tensorflow.keras.optimizers import SGD

from constants import *
from util import *
from models import *


def main(retrain: bool) -> None:
    global ARGS
    model_path = 'models/{}.bin'.format(ARGS.model_name)
    if not Path(model_path).exists() or retrain:
        X, y = load_data(train=True)
        assert X.shape[0] == y.shape[0]

        model = models[ARGS.model_name]
        model.summary()

        model.compile(
            loss='binary_crossentropy',
            metrics=['accuracy'],
            optimizer=SGD(lr=0.01, momentum=0.9, clipnorm=5.0))
        model.fit(
            X,
            y,
            validation_split=0.33,
            epochs=ARGS.epochs,
            batch_size=ARGS.batch_size)

        os.system("mkdir -p models")
        model.save(model_path)
        print("Model {} saved!".format(model_path))

    model = keras.models.load_model(model_path)
    print('Loaded model from disk.')

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
        "--retrain",
        action='store_true',
        help="Specify this option to not train and use the latest saved model")
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
        'model_name',
        type=str,
        help="Model name (e.g. cnnlstm)"
    )

    ARGS = parser.parse_args()
    retrain = ARGS.retrain
    main(retrain)
