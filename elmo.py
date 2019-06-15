import tensorflow_hub as hub
from keras.models import Model
import tensorflow as tf
from keras.engine.topology import Layer
import keras.backend as K
from sklearn.model_selection import train_test_split
import keras
import os
from keras.layers import Dense , Input, PReLU , Dropout
from keras_contrib.callbacks import CyclicLR
import numpy as np
import util

import constants



class ElmoEmbeddingLayer(Layer):
    def __init__(self, trainable=True, **kwargs):
        self.dimensions = 1024
        self.trainable = trainable
        super(ElmoEmbeddingLayer, self).__init__(**kwargs)


    def build(self, input_shape):
        self.elmo = hub.Module(
            'https://tfhub.dev/google/elmo/2',
            trainable=self.trainable,
            name="{}_module".format(self.name))

        self.trainable_weights += K.tf.trainable_variables(scope="^{}_module/.*".format(self.name))
        super(ElmoEmbeddingLayer, self).build(input_shape)


    def call(self, x, mask=None):
        result = self.elmo(K.squeeze(K.cast(x, tf.string), axis=1),
                           as_dict=True,
                           signature='default',
        )['default']
        return result


    def compute_mask(self, inputs, mask=None):
        return K.not_equal(inputs, '--PAD--')


    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.dimensions)



def gen_model():
    inputs = Input(shape=(1, ), name='input', dtype=tf.string)
    X = inputs

    X = ElmoEmbeddingLayer()(X)

    X = Dense(512, activation='relu')(X)
    X = Dropout(0.3)(X)
    X = Dense(1, activation='sigmoid')(X)

    model = Model(
        inputs=inputs,
        outputs=X,
        name='elmo'
    )

    return model


def main():
    K.clear_session()
    model = gen_model()

    model.summary()
    model.compile(
        loss='binary_crossentropy',
        metrics=['accuracy'],
        optimizer=keras.optimizers.Adam(
            lr=1e-3,
            epsilon=0.001,
        )
    )

    X, y = util.load_data(train=True, as_text=True)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=constants.TRAIN_TEST_SPLIT_PERCENTAGE)

    filepath = 'elmo' + "-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = keras.callbacks.ModelCheckpoint(
                    os.path.join('.', filepath),
                    monitor='val_acc',
                    verbose=1,
                    save_best_only=True,
                    save_weights_only=True,
                    mode='max')
    tensorboard = keras.callbacks.TensorBoard(
                    log_dir='./logs',
                    histogram_freq=1)
    callbacks_list = [checkpoint, tensorboard]

    model_name = 'elmo'

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=1,
        batch_size=64,
        callbacks=callbacks_list)

    model_path = os.path.join('models', f'{model_name}.bin')
    os.system("mkdir -p models")
    model.save(model_path)
    print("Model {} saved!".format(model_path))




if __name__ == '__main__':
    main()
