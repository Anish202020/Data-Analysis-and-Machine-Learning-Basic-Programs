import numpy as np
import pandas as pd

from matplotlib.pyplot import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn import *
from sklearn import linear_model
from sklearn.metrics import mean_squared_error,r2_score,accuracy_score
from sklearn import datasets,model_selection,svm
from sklearn.model_selection import train_test_split



X,Y = datasets.make_regression(n_samples=300,n_features=5,n_targets=1,random_state=0,noise=8.0)

X_train, X_test,Y_train, Y_test = model_selection.train_test_split(X,Y,test_size=0.20)

regr = linear_model.LinearRegression()
regr.fit(X_train,Y_train)

Y_pred = regr.predict(X_test)
Coef = regr.coef_
R2 = r2_score(Y_test,Y_pred)
MSE = mean_squared_error(Y_test,Y_pred)

print(Coef,R2,MSE)

style.use('ggplot')
plt.scatter(Y_test,Y_pred,color="green")
plt.title('Y_test vs Y_pred')
plt.xlabel('Y_pred')
plt.ylabel('Y_test')
plt.show()

fig1 =plt.figure()
ax = Axes3D(fig1)
ax.scatter(X_test[:,0],X_test[:,1],Y_test,color="blue")
ax.scatter(X_test[:,0],X_test[:,1],Y_pred,color="red")

plt.title('Y_test vs Y_pred')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()