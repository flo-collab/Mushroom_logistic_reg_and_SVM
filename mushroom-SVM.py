import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

raw_data = pd.read_csv('TP_2_datset_mushrooms.csv')

#Le données sont des lettres, on souhaite des données en chiffres 
labelencoder=LabelEncoder()
for col in raw_data.columns:
	raw_data[col] = labelencoder.fit_transform(raw_data[col])

print(raw_data.head())

#On separe notre jeu de données:
#Les features
X = raw_data.iloc[:,1:23]
#Les labels
y = raw_data.iloc[:,0]

#Separation train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33)



svm = LinearSVC()

params = { 'C': np.logspace(-3, 3, 7) }

gs_svm = GridSearchCV(svm, params, cv=10)
gs_svm.fit(X_train, y_train)

print(gs_svm.best_params_)