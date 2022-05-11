import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split

# chess_df = pd.read_csv("datasets/chess/kr-vs-kp.data")
# led_df = pd.read_csv("datasets/led/led-creator")
votes_df = pd.read_csv("datasets/votes/house-votes-84.data")
votes_df = votes_df.replace("n", 0).replace("y", 1).replace("?", -1)
X_train, X_test, y_train, y_test = train_test_split(votes_df.iloc[:, 1:], votes_df.iloc[:, 0])

# computer mutual info
mutual_info = mutual_info_classif(X_train, y_train)
m_df = pd.DataFrame({"vote": votes_df.columns[1:], "info": mutual_info})
print(m_df)