import os
import tensorflow as tf
from my import MyTuner, MyHyperModel
from data_train import x_train, y_train
from data_test import x_test, y_test

SEED = 42
tf.random.set_seed(SEED)
PROJECT_NAME = 's1'
DIR = os.path.dirname(os.path.abspath(__file__))
DIR_POINT_FILE = os.path.join(DIR, PROJECT_NAME, 'ep{epoch:03d}_loss{loss:.3f}_acc{accuracy:.3f}_vloss{val_loss:.3f}_vacc{val_accuracy:.3f}.h5')
DIR_TENSORBOARD = os.path.join(DIR, 'tbs1')

MONITOR = 'val_accuracy'
MONITOR_MAX = 'max'
LR_FACTOR = 0.3
LR_MIN = 1e-6
PATIENCE = 10
VERBOSE = 1

EPOCHS = 1000
BATCH_SIZE = 64
SHUFFLE = False

hypermodel = MyHyperModel(
    learning_rate=[1e-2, 1e-3, 1e-4],
    input_units=12,
    input_activ='swish',
    output_units=56,
    output_activ='softmax',
    dense_nums=(3, 6, 1),
    dense_units=(56, 60, 1),
    dense_activ=['swish', 'softplus', 'tanh'],
    compile_loss='categorical_crossentropy',
    compile_metrics=['accuracy']
)

tuner = MyTuner(
    hypermodel,
    objective=MONITOR,
    max_epochs=EPOCHS,
    factor=3,
    hyperband_iterations=3,
    # seed=seed,
    # tune_new_entries=False, # Prevents unlisted parameters from being tuned
    directory='kerastuner',
    project_name=PROJECT_NAME
)

tuner.reload()
tuner.search_space_summary()

tuner.search(
    x_train,
    y_train,
    validation_data=(x_test, y_test),
    batch_size=BATCH_SIZE,
    shuffle=SHUFFLE,
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint(filepath=DIR_POINT_FILE, monitor=MONITOR, mode=MONITOR_MAX, save_best_only=True, save_weights_only=False, verbose=VERBOSE),  # Only save the weights that correspond to the maximum validation accuracy.
        tf.keras.callbacks.EarlyStopping(monitor=MONITOR, mode=MONITOR_MAX, patience=PATIENCE, verbose=VERBOSE),  # If val_loss doesn't improve for a number of epochs set with 'patience' var training will stop to avoid overfitting.
        tf.keras.callbacks.ReduceLROnPlateau(monitor=MONITOR, mode=MONITOR_MAX, factor=LR_FACTOR, min_lr=LR_MIN, patience=PATIENCE//2, verbose=VERBOSE),  # Learning rate is reduced by 'lr_factor' if val_loss stagnates for a number of epochs set with 'patience/2' var.
        tf.keras.callbacks.TensorBoard(DIR_TENSORBOARD,  histogram_freq=1)
    ]
)

tuner.results_summary()
