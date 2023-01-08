import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
from sklearn import metrics

df = pd.read_csv("../data/processed/testREG.csv")

X = df.drop(['sal17'], axis=1)
y = np.log1p(df['sal17'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)

with open("gbrt.pickle", 'rb') as file_1:
    bestREGmodel = pickle.load(file_1)

print('Score REG model:', bestREGmodel.score(X_test, y_test))