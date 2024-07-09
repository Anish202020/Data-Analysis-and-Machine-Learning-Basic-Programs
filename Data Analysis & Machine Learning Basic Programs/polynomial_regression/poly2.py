from sklearn import *
from sklearn import datasets,model_selection,linear_model
from sklearn.metrics import mean_squared_error,r2_score,accuracy_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

from matplotlib.pyplot import *
import matplotlib.pyplot as plt

import numpy as np

X,Y=datasets.make_regression(n_samples=300,n_features=2,n_targets=1,random_state=0,noise=10)

poly = PolynomialFeatures(degree=3)
X = poly.fit_transform(X)

X_train,X_test,Y_train,Y_test = model_selection.train_test_split(X,Y,test_size=0.30)

reg = linear_model.LinearRegression()
reg.fit(X_train,Y_train)

Y_pred = reg.predict(X_test)
Coef = reg.coef_
print(Coef)

R2=r2_score(Y_test,Y_pred)
MSE= mean_squared_error(Y_test,Y_pred)
print(R2,MSE)

style.use('ggplot')
plt.scatter(Y_pred,Y_test,color='blue')
plt.title('Predicted Values vs Real Values')
plt.xlabel('Y_pred')
plt.ylabel('Y_test')

plt.show()
