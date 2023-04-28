from db1 import DB_1_DF
import datetime
import numpy as np
import pandas as pd

arr = []
for _, row in DB_1_DF.iterrows():
    s = str(row[0])
    index = int(s[-2:])
    minute_of_day = int(s[-6:-2])
    day_of_year = int(s[0:len(s) - 6])
    nums = '111112113114115116122123124125126133134135136144145146155156166222223224225226233234235236244245246255256266333334335336344345346355356366444445446455456466555556566666'[index * 3:index * 3+3]
    num1 = int(nums[0])
    num2 = int(nums[1])
    num3 = int(nums[2])
    sum = num1 + num2 + num3
    is_odd = int(sum % 2 == 1)
    is_big = int(sum > 10)
    year = 2023
    date = datetime.datetime.fromordinal(datetime.datetime(year, 1, 1).toordinal() + day_of_year - 1)
    time = datetime.timedelta(minutes=minute_of_day)
    month = date.month
    day = date.day
    week_of_year = date.isocalendar()[1]
    day_of_week = date.weekday()
    hour = time.seconds//3600
    minute = time.seconds//60 % 60
    arr.append([
        minute_of_day, year, month, day, hour, minute,
        day_of_year, week_of_year, day_of_week,
        num1, num2, num3,
        sum, is_odd, is_big,
        nums, index
    ])
arr = np.array(arr, dtype=np.int16)

COLUMNS = ['minute_of_day', 'year', 'month', 'day', 'hour', 'minute', 'day_of_year', 'week_of_year', 'day_of_week', 'num1', 'num2', 'num3', 'sum', 'is_odd', 'is_big', 'nums', 'index']

DB_2_DF = pd.DataFrame(arr, columns=COLUMNS)

print("\n#########################")
print(DB_2_DF)
print("#########################\n")
