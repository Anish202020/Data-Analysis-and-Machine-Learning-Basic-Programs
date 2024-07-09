from sklearn import *
from sklearn import datasets,model_selection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score,mean_squared_error,classification_report,confusion_matrix

from sklearn.datasets._samples_generator import make_blobs

import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import seaborn as sns

X,Y = make_blobs(n_samples=500,n_features=6,centers=9,shuffle=True,random_state=0)
X_train, X_test,Y_train,Y_test = model_selection.train_test_split(X,Y,test_size=.2)

sc = StandardScaler()
sc.fit(X_train)
X_train_std=sc.transform(X_train)
X_test_std = sc.transform(X_test)

logreg = linear_model.LogisticRegression()

logreg.fit(X_train_std,Y_train)
Y_pred = logreg.predict(X_test_std)

C_M=confusion_matrix(Y_test,Y_pred)
print(C_M)

R2= r2_score(Y_test,Y_pred)
C_R = classification_report(Y_test,Y_pred)
MSE  = mean_squared_error(Y_test,Y_pred)
print(R2,MSE,C_R)

plt.figure(figsize=(9,9))
sns.heatmap(C_M,annot=True,cmap='Blues_r')
plt.show()
