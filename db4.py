from db2 import DB_2_DF
from db3 import DB_3_DF
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

# DB_2_DF = StandardScaler().fit_transform(DB_2_DF)

x_train, x_test, y_train, y_test = train_test_split(DB_2_DF, DB_3_DF, test_size=0.2, random_state=42, shuffle=False)

print(f"x_train shape = {x_train.shape}, y_train shape = {y_train.shape}")

print(f"x_test shape = {x_test.shape}, y_test shape = {y_test.shape}")
