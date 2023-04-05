import os
import tensorflow as tf
from my import MyTuner, MyHyperModel
from data_train import x_train, y_train
from data_test import x_test, y_test

seed = 42
tf.random.set_seed(seed)

dir = os.path.dirname(os.path.abspath(__file__))
POINT_FILE = os.path.join(dir, 's1', 'ep{epoch:03d}_loss{loss:.3f}_acc{accuracy:.3f}_vloss{val_loss:.3f}_vacc{val_accuracy:.3f}.h5')
TENSORBOARD_DIR = os.path.join(dir, 'tbs1')

hypermodel = MyHyperModel(
    learning_rate=[1e-2, 1e-3, 1e-4],
    input_units=12,
    input_activ='swish',
    output_units=56,
    output_activ='softmax',
    dense_nums=(3, 6, 1),
    dense_units=(56, 60, 1),
    dense_activ=['swish', 'softplus', 'tanh']
)
tuner = MyTuner(
    hypermodel,
    objective='val_accuracy',
    max_epochs=1000,
    factor=3,
    hyperband_iterations=3,
    # seed=seed,
    # tune_new_entries=False, # Prevents unlisted parameters from being tuned
    directory='kerastuner',
    project_name='s1'
)
tuner.reload()
tuner.search_space_summary()
tuner.search(
    x_train,
    y_train,
    validation_data=(x_test, y_test),
    batch_size=64,
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint(filepath=POINT_FILE, monitor='val_accuracy', mode="max", save_best_only=True, save_weights_only=False),  # Only save the weights that correspond to the maximum validation accuracy.
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=5, verbose=1),  # If val_loss doesn't improve for a number of epochs set with 'patience' var training will stop to avoid overfitting.
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", mode="max", factor=0.3, min_lr=1e-6, patience=2, verbose=1),  # Learning rate is reduced by 'lr_factor' if val_loss stagnates for a number of epochs set with 'patience/2' var.
        tf.keras.callbacks.TensorBoard(TENSORBOARD_DIR,  histogram_freq=1)
    ]
)
tuner.results_summary()
