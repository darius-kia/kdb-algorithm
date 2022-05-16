import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from pyitlib import discrete_random_variable as drv
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder

# chess_df = pd.read_csv("datasets/chess/kr-vs-kp.data")
# led_df = pd.read_csv("datasets/led/led-creator")
votes_df = pd.read_csv("datasets/votes/house-votes-84.data")
votes_df = votes_df.replace("n", 0).replace("y", 1).replace("?", -1)
votes_df['class'] = votes_df['class'].map({"republican": 0, "democrat": 1})
# X_train, X_test, y_train, y_test = train_test_split(votes_df.iloc[:, 1:], votes_df.iloc[:, 0])

# computer mutual info
mutual_info = mutual_info_classif(votes_df.iloc[:, 1:], votes_df.iloc[:, 0])
m_df = pd.DataFrame({"vote": votes_df.columns[1:], "info": mutual_info})

# compute conditional mutual information
conditional_mutual_info = pd.DataFrame(data={i: [drv.information_mutual_conditional(votes_df[i], votes_df[j], votes_df.iloc[:, 0]) for j in votes_df.columns[1:]] for i in votes_df.iloc[:, 1:]}, index=votes_df.columns[1:])
print(conditional_mutual_info['handicapped-infants'].loc[['el-salvador-aid', 'mx-missile']])
# print(conditional_mutual_info)
# print(conditional_mutual_info['handicapped-infants'].astype(float).nlargest(3).index)
# print(drv.information_mutual_conditional(X_train['handicapped-infants'], X_train["crime"], y_train))
def join_attrs(k):
    order = m_df.sort_values(by="info", ascending=False)['vote']
    space = set()
    joins = []
    for attr in order:
        out = [attr]
        out += list(conditional_mutual_info[attr].loc[list(space)].astype(float).nlargest(k-1).index)
        # out += list(conditional_mutual_info[attr].astype(float).nlargest(k).index[1:])
        joins.append(sorted(out))
        space.add(attr)
    return joins
new_attrs = join_attrs(3)
new_df = pd.DataFrame()
# print(votes_df['physician-fee-freeze'])
for j in new_attrs:
    new_col = "_".join(j)
    new_df[new_col] = votes_df[j[0]].astype(str)
    for a in j[1:]:
        # votes_
        new_df[new_col] = new_df[new_col] + "_" + votes_df[a].astype(str)
# new_df.drop(columns="physician-fee-freeze", inplace=True)
new_df = new_df.apply(LabelEncoder().fit_transform)
print(new_df)
print(new_df.columns)
X_train, X_test, y_train, y_test = train_test_split(new_df, votes_df.iloc[:, 0])
gnb = GaussianNB()
print(X_train)
gnb.fit(X_train, y_train)
print(gnb.score(X_test, y_test))