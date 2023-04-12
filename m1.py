import tensorflow as tf
import keras_tuner as kt


class MyHP(kt.HyperParameters):
    pass


class MyTuner(kt.Hyperband):
    pass


class MyHyperModel(kt.HyperModel):
    def __init__(self, learning_rate, input, output, depth, lstm, dense, compile, stateful):
        self.learning_rate = learning_rate
        self.input = input
        self.output = output
        self.depth = depth
        self.lstm = lstm
        self.dense = dense
        self.compile = compile
        self.stateful = stateful

    def build(self, hp):
        '''['relu', 'sigmoid', 'softmax', 'softplus', 'softsign', 'tanh', 'selu', 'elu', 'exponential', 'leaky_relu', 'swish', 'exponential', 'hard_sigmoid', 'linear']'''
        model = tf.keras.Sequential()
        # INPUT ##########################################################################
        model.add(self.input)
        # DEPTH ##########################################################################
        hp_depth = hp.Int("depth", min_value=self.depth[0], max_value=self.depth[1], step=self.depth[2])
        for i in range(hp_depth):
            if i < hp_depth-1:
                model.add(tf.keras.layers.LSTM(
                    units=hp.Int(f"lstm_{i}", min_value=self.lstm[0], max_value=self.lstm[1], step=self.lstm[2]),
                    return_sequences=True,
                    stateful=self.stateful,
                    name=f"lstm_{i}"
                ))
                model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(
                    units=hp.Int(f"units_{i}", min_value=self.dense[0], max_value=self.dense[1], step=self.dense[2], default=self.dense[0]),
                    activation=hp.Choice(f"activ_{i}", self.dense[3]),
                    name=f"dense_{i}"
                )))
            else:
                model.add(tf.keras.layers.LSTM(
                    units=self.output[0],
                    return_sequences=False,
                    stateful=self.stateful,
                    name=f"lstm_{i}"
                ))
                model.add(tf.keras.layers.Dense(
                    units=self.output[0],
                    activation=hp.Choice(f"activ_{i}", self.dense[3]),
                    name=f"dense_{i}"
                ))
        # OUTPUT ##########################################################################
        model.add(tf.keras.layers.Dense(units=self.output[0], activation=self.output[1], name="output"))
        # COMPILE #########################################################################
        if self.compile[0] == 'adam':
            optimizer = tf.keras.optimizers.Adam(hp.Choice("learning_rate", values=self.learning_rate))
        model.compile(
            optimizer=optimizer,
            loss=self.compile[1],
            metrics=self.compile[2],
            run_eagerly=True
        )
        return model
