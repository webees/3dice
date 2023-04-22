from db4 import x_train, y_train, x_test, y_test
from db2 import COLUMNS
import numpy as np
import pandas as pd
from keras.preprocessing.sequence import TimeseriesGenerator

BATCH_SIZE = 1440
WINDOW_SIZE = 1439
FEATURES_NUM = x_train.shape[1]
LABLES_NUM = y_train.shape[1]

train = TimeseriesGenerator(x_train.values, y_train.values, length=WINDOW_SIZE, batch_size=BATCH_SIZE)
test = TimeseriesGenerator(x_test.values, y_test.values, length=WINDOW_SIZE, batch_size=BATCH_SIZE)
############################################################################################################################
train = [batch for batch in train if batch[0].shape[0] == BATCH_SIZE]
x_train = np.concatenate([batch[0] for batch in train])
y_train = np.concatenate([batch[1] for batch in train])

test = [batch for batch in test if batch[0].shape[0] == BATCH_SIZE]
x_test = np.concatenate([batch[0] for batch in test])
y_test = np.concatenate([batch[1] for batch in test])
############################################################################################################################
print(f"BATCH {len(train)}: x_train = {train[0][0].shape}, y_train = {train[0][1].shape}")
print(f"BATCH {len(test)}: x_test = {test[0][0].shape}, y_test = {test[0][1].shape}")
############################################################################################################################
print("\n#########################")
print(pd.DataFrame(train[0][0][0], columns=COLUMNS))
print(pd.DataFrame([train[0][1][0]], columns=COLUMNS))
print("#########################\n")
