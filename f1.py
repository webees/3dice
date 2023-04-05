import os
import tensorflow as tf
from data_train import x_train, y_train
from data_test import x_test, y_test

seed = 42
tf.random.set_seed(seed)

dir = os.path.dirname(os.path.abspath(__file__))
POINT_FILE = os.path.join(dir, 'hypermodel.h5')
TENSORBOARD_DIR = os.path.join(dir, 'tbf1')

MONITOR = 'val_accuracy'
MONITOR_MAX = 'max'
LR_FACTOR = 0.3
LR_MIN = 1e-6
PATIENCE = 10
VERBOSE = 1

EPOCHS=1000
BATCH_SIZE = 1

model = tf.keras.models.load_model('hypermodel.h5')

history = model.fit(
    x_train,
    y_train,
    validation_data=(x_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint(filepath=POINT_FILE, monitor=MONITOR, mode=MONITOR_MAX, save_best_only=True, save_weights_only=False),  # Only save the weights that correspond to the maximum validation accuracy.
        tf.keras.callbacks.EarlyStopping(monitor=MONITOR, mode=MONITOR_MAX, patience=PATIENCE, verbose=VERBOSE),  # If val_loss doesn't improve for a number of epochs set with 'patience' var training will stop to avoid overfitting.
        tf.keras.callbacks.ReduceLROnPlateau(monitor=MONITOR, mode=MONITOR_MAX, factor=LR_FACTOR, min_lr=LR_MIN, patience=PATIENCE, verbose=VERBOSE),  # Learning rate is reduced by 'lr_factor' if val_loss stagnates for a number of epochs set with 'patience/2' var.
        tf.keras.callbacks.TensorBoard(TENSORBOARD_DIR,  histogram_freq=1)
    ]
)

print(history.history)