import pandas as pd
import numpy as np


df = pd.read_csv('C:/Users/nidheesh/wdbc.data', header=None)


X = df.drop(columns=[0, 1]).values  # Assuming first column is ID, second is the label
y = df[1].map({'M': 1, 'B': 0}).values  # Mapping Malignant to 1, Benign to 0

from sklearn.covariance import EllipticEnvelope
ee = EllipticEnvelope(contamination=0.01)  # Adjust contamination as needed
y_pred = ee.fit_predict(X)
X_clean, y_clean = X[y_pred == 1, :], y[y_pred == 1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)

from xgboost import XGBClassifier
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

import matplotlib.pyplot as plt
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]  # Sort in decreasing order
plt.figure()
plt.title("Feature importances")
plt.bar(range(X_train.shape[1]), importances[indices])
plt.xticks(range(X_train.shape[1]), indices)
plt.show()