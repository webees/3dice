from db2 import DB_2_DF
import numpy as np
import pandas as pd

# arr = []
# for _, row in DB_2_DF.iterrows():
#     t = [0] * 56
#     t[row[15]] = 1
#     arr.append(t)
# arr = np.array(arr, dtype=int)

# DB_3_DF = pd.DataFrame(arr)

DB_3_DF = DB_2_DF

print(DB_3_DF)
print("\n\n")
