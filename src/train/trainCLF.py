import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.metrics import multilabel_confusion_matrix, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import make_pipeline

df = pd.read_csv("./testCLF.csv")

X = df.drop(['Active22'], axis=1)
y = df["Active22"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)

ada_pipe = make_pipeline(
        MinMaxScaler(), 
        AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth=1),
                             n_estimators=400,
                             learning_rate=0.22,
                             random_state=42)
        )
ada_pipe.fit(X_train, y_train)
predictions = ada_pipe.predict(X_test)

print('F1 Score:', f1_score(y_test, predictions))

print(metrics.confusion_matrix(y_test, predictions))

with open('../model/model_v0_clf.pickle', 'wb') as handle:
    pickle.dump(ada_pipe, handle, protocol=pickle.HIGHEST_PROTOCOL)