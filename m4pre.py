import os
import tensorflow as tf
import pandas as pd
import numpy as np
from keras.models import load_model
from db2 import COLUMNS
from db5 import x_test, y_test

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
###############################################################################################################################
tf.random.set_seed(42)
###############################################################################################################################
__PROJECT = os.path.splitext(os.path.basename(__file__))[0]
__DIR = os.path.dirname(os.path.abspath(__file__))
__POINT_H5 = os.path.join(__DIR, __PROJECT + '.h5')
###############################################################################################################################
print(pd.DataFrame(x_test[0], columns=COLUMNS))
print(y_test[0])
###############################################################################################################################
model = load_model(__POINT_H5)
predictions = model.predict(x_test[0:1])
prediction = np.array(predictions)
prediction = np.round(prediction).astype(int)
prediction = prediction.flatten()
print(prediction)

