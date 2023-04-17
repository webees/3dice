import os
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from m3db import x_train, y_train, x_test, y_test
###############################################################################################################################
SEED = 42
tf.random.set_seed(SEED)
###############################################################################################################################
__PROJECT = os.path.splitext(os.path.basename(__file__))[0]
__DIR = os.path.dirname(os.path.abspath(__file__))
__POINT_H5 = os.path.join(__DIR, __PROJECT + '.h5')
###############################################################################################################################
if os.path.exists(__POINT_H5):
    model = load_model(__POINT_H5)
else:
    model = Sequential()
    # 1
    model.add(Dense(units=11, activation="swish"))
    # 2
    model.add(Dense(units=58, activation="swish"))
    # 3
    model.add(Dense(units=60, activation="swish"))
    # 4
    model.add(Dense(units=57, activation="softplus"))
    # 5
    model.add(Dense(units=57, activation="softplus"))
    # 6
    model.add(Dense(units=56, activation="softmax"))
    model.compile(Adam(0.0001), loss="categorical_crossentropy", metrics=['accuracy'], run_eagerly=True)
###############################################################################################################################
history = model.fit(
    x_train,
    y_train,
    validation_data=(x_test, y_test),
    epochs=100,
    batch_size=64,
    shuffle=False,
    verbose=1,
    callbacks=[
        ModelCheckpoint(filepath=__POINT_H5, monitor='val_accuracy', mode='max', save_best_only=True, save_weights_only=False, verbose=1),
        EarlyStopping(monitor='val_accuracy', mode='max', patience=10, verbose=1),
        ReduceLROnPlateau(monitor='val_accuracy', mode='max', factor=0.1, min_lr=1e-6, patience=5, verbose=1),
        # TensorBoard(DIR_TENSORBOARD,  histogram_freq=1)
    ]
)

print(history.history)
