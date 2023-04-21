import sqlite3
import numpy as np
import pandas as pd

conn = sqlite3.connect('db.sqlite3')

cur = conn.cursor()

sql = cur.execute('SELECT * FROM game631').fetchall()

cur.close()
conn.close()

arr = np.array(sql, dtype=np.int64)
DB_1_DF = pd.DataFrame(arr, columns=['id'])

print("\n#########################")
print(DB_1_DF)
print("#########################\n")
