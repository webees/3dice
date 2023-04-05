import tensorflow as tf
import keras_tuner as kt


class MyHP(kt.HyperParameters):
    pass


class MyTuner(kt.Hyperband):
    pass


class MyHyperModel(kt.HyperModel):
    def __init__(self, learning_rate, input_units, input_activ, output_units, output_activ, dense_nums, dense_units, dense_activ, compile_loss, compile_metrics):
        self.learning_rate = learning_rate
        self.input_units = input_units
        self.input_activ = input_activ
        self.output_units = output_units
        self.output_activ = output_activ
        self.dense_nums = dense_nums
        self.dense_units = dense_units
        self.dense_activ = dense_activ
        self.compile_loss = compile_loss
        self.compile_metrics = compile_metrics

    def build(self, hp):
        '''
        Write a function that creates and returns a Keras model. Use the hp argument to define the hyperparameters during model creation.
        ['relu', 'sigmoid', 'softmax', 'softplus', 'softsign', 'tanh', 'selu', 'elu', 'exponential', 'leaky_relu', 'swish', 'exponential', 'hard_sigmoid', 'linear']
        '''
        hp_dense_nums = hp.Int("dense_nums", min_value=self.dense_nums[0], max_value=self.dense_nums[1], step=self.dense_nums[2])
        hp_lr = hp.Choice("learning_rate", values=self.learning_rate)
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(units=self.input_units, activation=self.input_activ))  # Intput layer.
        for i in range(hp_dense_nums):  # Number of layers of the MLP is a hyperparameter.
            _units = hp.Int(f"units_{i}", min_value=self.dense_units[0], max_value=self.dense_units[1], step=self.dense_units[2])  # Number of units of each layer are different hyperparameters with different names.
            _activ = hp.Choice(f"activ_{i}", self.dense_activ)
            model.add(tf.keras.layers.Dense(units=_units, activation=_activ))
        model.add(tf.keras.layers.Dense(units=self.output_units, activation=self.output_activ))  # Output layer.
        model.compile(
            optimizer=tf.keras.optimizers.Adam(hp_lr),  # Automatically chooses learning rate.
            loss=self.compile_loss,
            metrics=self.compile_metrics,
            run_eagerly=True
        )
        return model
