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

df = pd.read_csv('temperature.csv')


year=df.iloc[0:,3]
print(year)

X1 = df.iloc[0:,4]

X2= df.iloc[0:,5]

#Average
Y = df.iloc[0:,1]
print(Y)
fig = plt.figure()
ax=Axes3D(fig)
ax.scatter(X1,X2,Y)
plt.show()

X = pd.concat([X1,X2],axis=1)

X_train, X_test, Y_train, Y_test=model_selection.train_test_split(X,Y,test_size=0.3)

reg = linear_model.LinearRegression()

reg.fit(X_train,Y_train)

Y_pred=reg.predict(X_test)
Coef = reg.coef_
R2 = r2_score(Y_test,Y_pred)
MSE = mean_squared_error(Y_test, Y_pred)
print(Coef,R2,MSE)

style.use('ggplot')
plt.scatter(Y_test,Y_pred,color='blue')
plt.title('Predicted Data Vs Real Data')
plt.show()

fig2 = plt.figure()
ax2 = Axes3D(fig2)
ax.scatter(X_test['LandMaxTemperature'],X_test['LandMinTemperature'],Y_test,color='red')
ax.scatter(X_test['LandMaxTemperature'],X_test['LandMinTemperature'],Y_pred,color='green')
plt.show()
