import os
import tensorflow as tf
from data_train import x_train, y_train
from data_test import x_test, y_test

SEED = 42
tf.random.set_seed(SEED)
DIR = os.path.dirname(os.path.abspath(__file__))
DIR_POINT_FILE = os.path.join(DIR, 'hypermodel.h5')
DIR_TENSORBOARD = os.path.join(DIR, 'tbf1')

MONITOR = 'val_accuracy'
MONITOR_MAX = 'max'
LR_FACTOR = 0.3
LR_MIN = 1e-6
PATIENCE = 10
VERBOSE = 1

EPOCHS = 1000
BATCH_SIZE = 1
SHUFFLE = False

model = tf.keras.models.load_model('hypermodel.h5')

history = model.fit(
    x_train,
    y_train,
    validation_data=(x_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=SHUFFLE,
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint(filepath=DIR_POINT_FILE, monitor=MONITOR, mode=MONITOR_MAX, save_best_only=True, save_weights_only=False),
        tf.keras.callbacks.EarlyStopping(monitor=MONITOR, mode=MONITOR_MAX, patience=PATIENCE, verbose=VERBOSE),
        tf.keras.callbacks.ReduceLROnPlateau(monitor=MONITOR, mode=MONITOR_MAX, factor=LR_FACTOR, min_lr=LR_MIN, patience=PATIENCE, verbose=VERBOSE),
        tf.keras.callbacks.TensorBoard(DIR_TENSORBOARD,  histogram_freq=1)
    ]
)

print(history.history)
