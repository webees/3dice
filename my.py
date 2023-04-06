import tensorflow as tf
import keras_tuner as kt


class MyHP(kt.HyperParameters):
    pass


class MyTuner(kt.Hyperband):
    pass


class MyHyperModel(kt.HyperModel):
    def __init__(self, learning_rate, input, output, depth, lstm,  dense, compile):
        self.learning_rate = learning_rate
        self.input = input
        self.output = output
        self.depth = depth
        self.lstm = lstm
        self.dense = dense
        self.compile = compile

    def build(self, hp):
        '''['relu', 'sigmoid', 'softmax', 'softplus', 'softsign', 'tanh', 'selu', 'elu', 'exponential', 'leaky_relu', 'swish', 'exponential', 'hard_sigmoid', 'linear']'''
        hp_depth = hp.Int("depth", min_value=self.depth[0], max_value=self.depth[1], step=self.depth[2])
        hp_lr = hp.Choice("learning_rate", values=self.learning_rate)
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(units=self.input[0], activation=self.input[1]))
        for i in range(hp_depth):
            lstm_units = hp.Int(f"lstm_{i}", min_value=self.lstm[0], max_value=self.lstm[1], step=self.lstm[2])
            lstm_activ = hp.Choice(f"lstm_{i}", self.lstm[3])
            dense_units = hp.Int(f"units_{i}", min_value=self.dense[0], max_value=self.dense[1], step=self.dense[2])
            dense_activ = hp.Choice(f"activ_{i}", self.dense[3])
            model.add(tf.keras.layers.LSTM(units=lstm_units, activation=lstm_activ))
            model.add(tf.keras.layers.Dense(units=dense_units, activation=dense_activ))
        model.add(tf.keras.layers.Dense(units=self.output[0], activation=self.output[1]))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(hp_lr),
            loss=self.compile[0],
            metrics=self.compile[1],
            run_eagerly=True
        )
        return model
