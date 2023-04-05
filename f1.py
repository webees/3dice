import os
import tensorflow as tf
from data_train import x_train, y_train
from data_test import x_test, y_test

seed = 42
tf.random.set_seed(seed)

dir = os.path.dirname(os.path.abspath(__file__))
POINT_FILE = os.path.join(dir, 'hypermodel.h5')
TENSORBOARD_DIR = os.path.join(dir, 'tbf1')

model = tf.keras.models.load_model('hypermodel.h5')

history = model.fit(
    x_train,
    y_train,
    validation_data=(x_test, y_test),
    batch_size=1,
    epochs=1000,
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint(filepath=POINT_FILE, monitor='val_accuracy', mode="max", save_best_only=True, save_weights_only=False),  # Only save the weights that correspond to the maximum validation accuracy.
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=10, verbose=1),  # If val_loss doesn't improve for a number of epochs set with 'patience' var training will stop to avoid overfitting.
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", mode="max", factor=0.3, min_lr=1e-6, patience=5, verbose=1),  # Learning rate is reduced by 'lr_factor' if val_loss stagnates for a number of epochs set with 'patience/2' var.
        tf.keras.callbacks.TensorBoard(TENSORBOARD_DIR,  histogram_freq=1)
    ]
)
