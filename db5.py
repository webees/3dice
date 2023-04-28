from db4 import x_train, y_train, x_test, y_test
from db2 import COLUMNS
import numpy as np
import pandas as pd
from keras.preprocessing.sequence import TimeseriesGenerator

BATCH_SIZE = 360
WINDOW_SIZE = 359
FEATURES_NUM = x_train.shape[1]
LABLES_NUM = 1
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
for i, batch in enumerate(train):
    x, y = batch
    print(f"BATCH {i}: x.shape={x.shape}, y.shape={y.shape}")
############################################################################################################################
print("\n#########################")
print(pd.DataFrame(x_train[42], columns=COLUMNS))
print(pd.DataFrame(y_train, columns=COLUMNS))
############################################################################################################################
y_train = pd.DataFrame(y_train, columns=COLUMNS)
y_train = y_train['is_odd']
y_train = y_train.values
############################################################################################################################
print(pd.DataFrame(y_train, columns=['is_odd']))
print("\n#########################")
