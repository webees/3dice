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
        model.add(self.input)
        for i in range(hp_depth):
            lstm_units = hp.Int(f"lstm_{i}", min_value=self.lstm[0], max_value=self.lstm[1], step=self.lstm[2])
            dense_units = hp.Int(f"units_{i}", min_value=self.dense[0], max_value=self.dense[1], step=self.dense[2])
            dense_activ = hp.Choice(f"activ_{i}", self.dense[3])
            model.add(tf.keras.layers.LSTM(name=f"lstm_{i}", units=lstm_units, return_sequences = True, stateful=True))
            model.add(tf.keras.layers.Dense(name=f"dense_{i}", units=dense_units, activation=dense_activ))
        model.add(self.output)
        if self.compile[0] == 'adam':
            optimizer = tf.keras.optimizers.Adam(hp_lr)
        model.compile(
            optimizer=optimizer,
            loss=self.compile[1],
            metrics=self.compile[2],
            run_eagerly=True
        )
        return model
