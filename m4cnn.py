import os
import tensorflow as tf
from keras.models import Sequential, load_model
import pandas as pd
from keras.layers import CuDNNLSTM, Dense, BatchNormalization, Flatten, TimeDistributed, LayerNormalization, Dropout, Conv1D, MaxPooling1D, ReLU, GRU, InputLayer, Input
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras.optimizers import Adagrad, Adam
from db5 import x_train, x_val, y_train, y_val, x_test, y_test, BATCH_SIZE, WINDOW_SIZE, FEATURES_NUM, LABLES_NUM
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
    model.add(InputLayer(batch_input_shape=(BATCH_SIZE, WINDOW_SIZE, FEATURES_NUM), name="input"))
    model.add(Conv1D(filters=FEATURES_NUM*4, kernel_size=3, activation='swish', name="conv1d_1"))
    model.add(MaxPooling1D(pool_size=2, name="max_pooling1d_1"))
    model.add(Conv1D(filters=FEATURES_NUM*4, kernel_size=3, activation='swish', name="conv1d_2"))
    model.add(MaxPooling1D(pool_size=2, name="max_pooling1d_2"))
    model.add(Flatten(name="flatten_1"))
    model.add(Dense(units=FEATURES_NUM*8, activation='swish', name="dense_1"))
    model.add(Dropout(0.3, name="dropout_1"))
    model.add(Dense(units=FEATURES_NUM*4, activation='swish', name="dense_2"))
    model.add(Dropout(0.3, name="dropout_2"))
    model.add(Dense(units=LABLES_NUM, activation='linear', name="output"))
    model.compile(optimizer=Adagrad(0.0001), loss="mse", metrics=['mse', 'mae', 'accuracy'], run_eagerly=True)
###############################################################################################################################
model.build(input_shape=(BATCH_SIZE, WINDOW_SIZE, FEATURES_NUM))
model.compile(optimizer=Adagrad(0.0001), loss="mse", metrics=['mse', 'mae', 'accuracy'], run_eagerly=True)
###############################################################################################################################
model.summary()
for i, layer in enumerate(model.layers):
    print("Layer Index: {}, Layer Name: {}".format(i, layer.name))
###############################################################################################################################
# loss, mse, acc = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)
# print(f"Validation Loss: {loss:.4f}, Validation MSE: {mse:.4f}, Validation Accuracy: {acc:.4f}")
###############################################################################################################################
history = model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=1000,
    batch_size=BATCH_SIZE,
    verbose=1,
    shuffle=False,
    callbacks=[
        ModelCheckpoint(filepath=__POINT_H5, monitor='mse', mode='min', save_best_only=True, save_weights_only=False, verbose=1),
        ReduceLROnPlateau(monitor='mse', mode='min', factor=0.3, min_lr=1e-6, patience=5, verbose=1),
    ]
)

print(history.history)
