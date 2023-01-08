import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
import pickle
import numpy as np
from sklearn import metrics

from sklearn.pipeline import make_pipeline

df = pd.read_csv("./testREG.csv")

X = df.drop(['sal17'], axis=1)
y = np.log1p(df['sal17'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)

gbrt_pipe = make_pipeline(MinMaxScaler(), GradientBoostingRegressor(max_depth=3,
                                n_estimators=50,
                                learning_rate=0.3,
                                random_state=42
								))
gbrt_pipe.fit(X_train, y_train)

print('Score REG model:', gbrt_pipe.score(X_test, y_test))

with open('../model/model_v0_reg.pickle', 'wb') as handle:
    pickle.dump(gbrt_pipe, handle, protocol=pickle.HIGHEST_PROTOCOL)