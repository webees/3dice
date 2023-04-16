import keras_tuner as kt
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import LSTM, Dense, TimeDistributed, Dropout


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
        model = Sequential()
        # INPUT ##########################################################################
        model.add(self.input)
        # DEPTH ##########################################################################
        hp_depth = hp.Int("depth", min_value=self.depth[0], max_value=self.depth[1], step=self.depth[2])
        for i in range(hp_depth):
            if i < hp_depth-1:
                model.add(LSTM(
                    units=hp.Int(f"{i}_lstm", min_value=self.lstm[0], max_value=self.lstm[1], step=self.lstm[2]),
                    dropout=hp.Float(f"{i}_lstm__drop",min_value=0,max_value=0.5,step=0.1),
                    return_sequences=True,
                    stateful=self.stateful,
                    name=f"{i}_lstm"
                ))
                model.add(Dropout(hp.Float(f"{i}_drop",min_value=0,max_value=0.5,step=0.1)))
                model.add(TimeDistributed(Dense(
                    units=hp.Int(f"{i}_units", min_value=self.dense[0], max_value=self.dense[1], step=self.dense[2], default=self.dense[0]),
                    activation=hp.Choice(f"{i}_activ", self.dense[3]),
                    name=f"{i}_dense"
                )))
            else:
                model.add(LSTM(
                    units=self.output[0],
                    dropout=hp.Float(f"{i}_lstm_drop",min_value=0,max_value=0.5,step=0.1),
                    return_sequences=False,
                    stateful=self.stateful,
                    name=f"{i}_lstm"
                ))
                model.add(Dropout(hp.Float(f"{i}_drop",min_value=0,max_value=0.5,step=0.1)))
                model.add(Dense(
                    units=self.output[0],
                    activation=hp.Choice(f"{i}_activ", self.dense[3]),
                    name=f"dense_{i}"
                ))
        # OUTPUT ##########################################################################
        model.add(Dense(units=self.output[0], activation=self.output[1], name="output"))
        # COMPILE #########################################################################
        if self.compile[0] == 'adam':
            optimizer = Adam(hp.Choice("learning_rate", values=self.learning_rate))
        model.compile(
            optimizer=optimizer,
            loss=self.compile[1],
            metrics=self.compile[2],
            run_eagerly=True
        )
        return model
