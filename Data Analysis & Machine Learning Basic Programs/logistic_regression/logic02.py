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

from mlxtend.plotting import plot_decision_regions

iris = datasets.load_iris()
X=iris.data[:,[1,3]]
Y=iris.target
# 0 sepal length
# 1 sepal width
# 3 petal length
# 4 petal width cm
# 5 class Setosa Versicolor Virginica

X_train,X_test,Y_train,Y_test = model_selection.train_test_split(X,Y,test_size=.25)
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

logreg = linear_model.LogisticRegression()
logreg.fit(X_train_std,Y_train)

Y_pred = logreg.predict(X_test_std)

C_M = confusion_matrix(Y_test,Y_pred)
C_R = classification_report(Y_test,Y_pred)
R2 = r2_score(Y_test,Y_pred)
MSE = mean_squared_error(Y_test,Y_pred)

print(C_M,C_R,R2,MSE)

plot_decision_regions(X_test_std,Y_pred,clf=logreg,legend=2)
plt.title('Logistic Regression on IRIS Dataset')
plt.xlabel('sepal width')
plt.ylabel('pepal width')
plt.show()

