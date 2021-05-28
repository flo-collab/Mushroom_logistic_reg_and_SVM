import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

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


lr = LogisticRegression(solver='liblinear')

lr.fit(X_train, y_train)

# on recupere nos predictions 
y_prob = lr.predict_proba(X_test)[:,1]

#On créé un vecteur prediction 0 ou 1 à partir du vecteur probabilités
y_pred = np.where(y_prob>0.5, 1, 0)

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(false_positive_rate, true_positive_rate)
print('the area under roc curve is :',roc_auc)

#graphique de cette courbe
plt.figure(figsize=(10,10))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],linestyle='--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
#plt.show()

#on essaye d'ameliorer la regression en testant d'autres parametres
lr = LogisticRegression(solver = 'liblinear')
params = {'C': np.logspace(-3, 3, 7) , 'penalty':['l1','l2'] }

lr_gs = GridSearchCV(lr, params, cv=10)
lr_gs.fit(X_train, y_train)

print(lr_gs.best_params_)

# On récupère la prédiction de la valeur positive
y_prob = lr_gs.predict_proba(X_test)[:,1] 

# On créé un vecteur de prédiction à partir du vecteur de probabilités
y_pred = np.where(y_prob > 0.5, 1, 0) 

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(false_positive_rate, true_positive_rate)
print(roc_auc)

#roc_auc up to 0.98 
