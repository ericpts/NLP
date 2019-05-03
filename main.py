#!/usr/bin/env python3
import os
import argparse
import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split

from constants import *
from util import *
from models import *


def main(model_name: str, retrain: bool) -> None:
    model_path = 'models/{}.bin'.format(model_name)
    if not Path(model_path).exists() or retrain:
        X, y = load_data(train=True)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=TRAIN_TEST_SPLIT_PERCENTAGE)

        assert X_train.shape[0] == y_train.shape[0]
        assert X_val.shape[0] == y_val.shape[0]
        print('Train data: {}, Validation data: {}'.format(X_train.shape[0], X_val.shape[0]))

        model = models[model_name]
        model.summary()

        model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, validation_data=(X_val, y_val))

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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--retrain",
        action='store_true',
        help="Specify this option to not train and use the latest saved model")
    parser.add_argument(
        'model',
        type=str,
        help="Model name (e.g. twitter_model1)"
    )

    args = parser.parse_args()

    retrain    = args.retrain
    model_name = args.model

    main(model_name, retrain)
