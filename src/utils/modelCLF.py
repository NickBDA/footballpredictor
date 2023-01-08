import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.metrics import multilabel_confusion_matrix, f1_score

df = pd.read_csv("../data/processed/testCLF.csv")

X = df.drop(['Active22'], axis=1)
y = df["Active22"]

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)

with open("ada_clf_F1.pickle", 'rb') as file_1:
    bestCLFmodel = pickle.load(file_1)

predictions = bestCLFmodel.predict(X_test)
print('F1 Score:', f1_score(y_test, predictions))

print(metrics.confusion_matrix(y_test, predictions))