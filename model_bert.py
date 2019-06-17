import os
os.environ['TF_KERAS'] = '1'
import keras_bert
import tensorflow as tf
import tensorflow.keras as keras
from pathlib import Path
import codecs
from sklearn.model_selection import train_test_split
import numpy as np
import util
import constants

BATCH_SIZE = 64
EPOCHS = 1

def get_bert_model_dir() -> Path:
    model_dir = ".models/uncased_L-12_H-768_A-12"
    return Path(model_dir)


def get_bert_model():
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

    inputs = indices
    X = bert_model(
        [inputs, tf.zeros_like(inputs)]
    )
    X = keras.layers.Dense(1, activation='sigmoid')(X)

    model = keras.models.Model(
        inputs=indices,
        outputs=X,
    )

    return model


def get_bert_data(train: bool):
    cache_f = Path(f'datasets/bert_{train}.npz')
    if cache_f.exists():
        data = np.load(str(cache_f))
        return data['X'], data['y']

    print('Computing bert data...')

    vocab_path = get_bert_model_dir() / 'vocab.txt'

    token_dict = {}
    with codecs.open(str(vocab_path), 'r', 'utf8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)
    token_dict['<user>'] = len(token_dict)
    token_dict['<url>'] = len(token_dict)

    tokenizer = keras_bert.Tokenizer(token_dict)
    X, y = util.load_data(train, as_text=True)
    X = [tokenizer.encode(x, max_len=constants.MAX_SEQUENCE_LENGTH)[0] for x in X]
    X = np.array(X)

    np.savez(
        str(cache_f),
        X=X,
        y=y
    )

    return X, y


def main():
    model = get_bert_model()
    model.summary()

    X, y = get_bert_data(True)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=constants.TRAIN_TEST_SPLIT_PERCENTAGE)

    decay_steps, warmup_steps = keras_bert.calc_train_steps(
        y_train.shape[0],
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
    )

    model.compile(
        optimizer=keras_bert.AdamWarmup(
            decay_steps=decay_steps, warmup_steps=warmup_steps, lr=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )

    filepath = 'bert' + "-{epoch:02d}-{val_acc:.2f}.hdf5"

    checkpoint = keras.callbacks.ModelCheckpoint(
        os.path.join('.', filepath),
        monitor='val_acc',
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode='max',
    )

    tensorboard = keras.callbacks.TensorBoard(
        log_dir='./logs',
    )

    callbacks_list = [
        checkpoint,
        # tensorboard,
    ]

    model_name = 'bert'

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks_list,
    )

    model_path = os.path.join('models', f'{model_name}.bin')
    os.system("mkdir -p models")
    model.save(model_path)
    print("Model {} saved!".format(model_path))




if __name__ == '__main__':
    main()
