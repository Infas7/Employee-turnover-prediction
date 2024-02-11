import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import pickle

# run this script to train the model


df = pd.read_csv('Employee Turnover Dataset.csv')
df.drop_duplicates(inplace=True)
df.rename(columns={"sales": "department", "salary": "salary_level"}, inplace=True)

categorical_cols = ["department", "salary_level"]
encoded_cols = pd.get_dummies(df[categorical_cols], prefix="cat")

df = df.join(encoded_cols)

df.drop(["department", "salary_level"], inplace=True, axis="columns")

X = df.drop("left", axis=1)
y = df["left"]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)



from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

logReg = LogisticRegression(max_iter=2000)

logReg.fit(X_train, y_train)
logReg_predictions = logReg.predict(X_test)

print(accuracy_score(y_test, logReg_predictions))
print(classification_report(y_test, logReg_predictions))



from sklearn.ensemble import RandomForestClassifier

ranForest = RandomForestClassifier(n_estimators=100)

ranForest.fit(X_train, y_train)
ranForest_predictions = ranForest.predict(X_test)

print(accuracy_score(y_test, ranForest_predictions))
print(classification_report(y_test, ranForest_predictions))


with open("model.pkl", "wb") as model_file:
  pickle.dump(ranForest, model_file)

