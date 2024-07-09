from sklearn import *
from sklearn import datasets,model_selection,linear_model
from sklearn.metrics import mean_squared_error,r2_score,accuracy_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

from matplotlib.pyplot import *
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

df = pd.read_csv('CO2vsTEMP.csv')
CO2=df.iloc[2:,2]

Temp=df.iloc[2:,1]
print(Temp.head())

year=df.iloc[2:,0]

style.use('ggplot')

plt.figure(0)
plt.plot(year,CO2,'red')
plt.title('CO2(ppm) vs Year')
plt.xlabel('Year')
plt.ylabel('CO2(ppm)')
plt.show(block=False)

plt.figure(1)
plt.plot(year,Temp,'blue')
plt.title('Temp vs Year')
plt.xlabel('Year')
plt.ylabel('Temp Farenhiet')
plt.show(block=False)

X=CO2.values.reshape(-1,1)
Y=Temp

poly=PolynomialFeatures(degree=3)
X=poly.fit_transform(X)

X_train ,X_test,Y_train,Y_test = model_selection.train_test_split(X,Y,test_size=0.25)

reg = linear_model.LinearRegression()
reg.fit(X_train,Y_train)

Y_pred=reg.predict(X_test)
# A*X^3 + B*X^2 + C*X + D

Coef=reg.coef_
print(Coef)

R2 = r2_score(Y_test,Y_pred)
MSE= mean_squared_error(Y_test,Y_pred)
print(R2,MSE)

plt.figure(2)
plt.scatter(Y_pred,Y_test,color='green')
plt.title('Y_pred vs Y_test')
plt.xlabel('Y_pred')
plt.ylabel('Y_test')
plt.show(block=False)

a= np.arange(0,len(Y_test))

plt.figure(3)
plt.scatter(a,Y_test,color='blue')
plt.scatter(a,Y_pred,color='red',marker='p')
plt.title('Y_pred and Y_test')
plt.xlabel('#')
plt.ylabel('Y_test & Y_pred')
plt.show()