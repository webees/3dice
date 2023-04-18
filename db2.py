from db1 import DB_1_DF
import datetime
import numpy as np
import pandas as pd

CODE_MAP = {'111': 0, '112': 1, '113': 2, '114': 3, '115': 4, '116': 5,
            '122': 6, '123': 7, '124': 8, '125': 9, '126': 10,
            '133': 11, '134': 12, '135': 13, '136': 14,
            '144': 15, '145': 16, '146': 17,
            '155': 18, '156': 19,
            '166': 20,
            '222': 21, '223': 22, '224': 23, '225': 24, '226': 25,
            '233': 26, '234': 27, '235': 28, '236': 29,
            '244': 30, '245': 31, '246': 32,
            '255': 33, '256': 34,
            '266': 35,
            '333': 36, '334': 37, '335': 38, '336': 39,
            '344': 40, '345': 41, '346': 42,
            '355': 43, '356': 44,
            '366': 45,
            '444': 46, '445': 47, '446': 48,
            '455': 49, '456': 50,
            '466': 51,
            '555': 52, '556': 53,
            '566': 54,
            '666': 55}

arr = []
for _, row in DB_1_DF.iterrows():
    s = str(row[0])
    year = int('20' + s[:2])
    month = int(s[2:4])
    day = int(s[4:6])
    hour = int(s[6:8])
    minute = int(s[8:10])
    num1 = int(s[10])
    num2 = int(s[11])
    num3 = int(s[12])
    dt = datetime.datetime(year, month, day, hour, minute)
    day_of_year = dt.timetuple().tm_yday
    week_of_year = dt.isocalendar()[1]
    day_of_week = dt.weekday()
    sum = num1 + num2 + num3
    is_odd = int(sum % 2 == 1)
    is_big = int(sum > 10)
    nums = s[10] + s[11] + s[12]
    index = CODE_MAP[nums]
    arr.append(
        [
            year, month, day, hour, minute,
            day_of_year, week_of_year, day_of_week,
            num1, num2, num3,
            sum, is_odd, is_big,
            nums, index
        ])
arr = np.array(arr, dtype=np.int64)

COLUMNS = ['year', 'month', 'day', 'hour', 'minute', 'day_of_year', 'week_of_year', 'day_of_week', 'num1', 'num2', 'num3', 'sum', 'is_odd', 'is_big', 'nums', 'index']

DB_2_DF = pd.DataFrame(arr, columns=COLUMNS)

print(DB_2_DF)
print("\n\n")