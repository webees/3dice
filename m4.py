import os
import tensorflow as tf
from keras.models import Sequential, load_model
import pandas as pd
from keras.layers import CuDNNLSTM, Dense, BatchNormalization, Flatten, TimeDistributed, LayerNormalization, Dropout, Conv1D, MaxPooling1D, ReLU, GRU, InputLayer, Input
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras.optimizers import Adagrad, Adam
from sklearn.model_selection import train_test_split
from db5 import x_train, x_test, y_train, y_test, BATCH_SIZE, WINDOW_SIZE, FEATURES_NUM, LABLES_NUM
###############################################################################################################################
tf.random.set_seed(42)
###############################################################################################################################
__PROJECT = os.path.splitext(os.path.basename(__file__))[0]
__DIR = os.path.dirname(os.path.abspath(__file__))
__POINT_H5 = os.path.join(__DIR, __PROJECT + '.h5')
###############################################################################################################################
if os.path.exists(__POINT_H5):
    model = load_model(__POINT_H5)
else:
    model = Sequential()
    model.add(InputLayer(batch_input_shape=(BATCH_SIZE, WINDOW_SIZE, FEATURES_NUM)))

    model.add(Conv1D(filters=FEATURES_NUM*2, kernel_size=3, activation='swish'))
    model.add(BatchNormalization())
    model.add(Conv1D(filters=FEATURES_NUM*2, kernel_size=3, activation='swish'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))

    model.add(Conv1D(filters=FEATURES_NUM*4, kernel_size=3, activation='swish'))
    model.add(BatchNormalization())
    model.add(Conv1D(filters=FEATURES_NUM*4, kernel_size=3, activation='swish'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))

    model.add(CuDNNLSTM(units=FEATURES_NUM*4, return_sequences=True, stateful=False))
    model.add(BatchNormalization())
    model.add(CuDNNLSTM(units=FEATURES_NUM*4, return_sequences=True, stateful=False))
    model.add(BatchNormalization())
    model.add(CuDNNLSTM(units=FEATURES_NUM*4, return_sequences=False, stateful=False))
    model.add(BatchNormalization())

    model.add(Dense(units=LABLES_NUM, activation="softmax"))
    model.compile(optimizer=Adagrad(0.0001), loss="categorical_crossentropy", metrics=['accuracy'], run_eagerly=True)
###############################################################################################################################
history = model.fit(
    x_train,
    y_train,
    validation_data=(x_test, y_test),
    epochs=10,
    batch_size=BATCH_SIZE,
    verbose=1,
    shuffle=False,
    callbacks=[
        ModelCheckpoint(filepath=__POINT_H5, monitor='val_accuracy', mode='max', save_best_only=True, save_weights_only=False, verbose=1),
        ReduceLROnPlateau(monitor='val_accuracy', mode='max', factor=0.3, min_lr=1e-6, patience=5, verbose=1),
    ]
)

print(history.history)
