import os
import tensorflow as tf
from m3db import x_train, y_train, x_test, y_test
###############################################################################################################################
SEED = 42
tf.random.set_seed(SEED)
###############################################################################################################################
DIR_PROJECT = 'm3'
DIR = os.path.dirname(os.path.abspath(__file__))
DIR_POINT_FILE = os.path.join(DIR, DIR_PROJECT + '.h5')
DIR_TENSORBOARD = os.path.join(DIR, DIR_PROJECT)
###############################################################################################################################
if os.path.exists(DIR_POINT_FILE):
    model = tf.keras.models.load_model(DIR_POINT_FILE)
else:
    model = tf.keras.Sequential()
    # 1
    model.add(tf.keras.layers.Dense(units=11, activation="swish"))
    # 2
    model.add(tf.keras.layers.Dense(units=58, activation="swish"))
    # 3
    model.add(tf.keras.layers.Dense(units=60, activation="swish"))
    # 4
    model.add(tf.keras.layers.Dense(units=57, activation="softplus"))
    # 5
    model.add(tf.keras.layers.Dense(units=57, activation="softplus"))
    # 6
    model.add(tf.keras.layers.Dense(units=56, activation="softmax"))
    model.compile(optimizer=tf.keras.optimizers.Adam(0.0001), loss="categorical_crossentropy", metrics=['accuracy'], run_eagerly=True)
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
        tf.keras.callbacks.ModelCheckpoint(filepath=DIR_POINT_FILE, monitor='val_accuracy', mode='max', save_best_only=True, save_weights_only=False, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', patience=10, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', mode='max', factor=0.1, min_lr=1e-6, patience=5, verbose=1),
        tf.keras.callbacks.TensorBoard(DIR_TENSORBOARD,  histogram_freq=1)
    ]
)

print(history.history)
