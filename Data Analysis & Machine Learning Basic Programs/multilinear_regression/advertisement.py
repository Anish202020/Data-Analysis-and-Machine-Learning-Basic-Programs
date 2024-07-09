import numpy as np
import pandas as pd

from matplotlib.pyplot import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn import *
from sklearn import linear_model
from sklearn.metrics import mean_squared_error,r2_score,accuracy_score


df = pd.read_csv('advertisement.csv')

TV = df.iloc[:,1]
print(TV)
radio = df.iloc[:,2]
print(radio)
newspaper = df.iloc[:,3]
print(newspaper)

sales = df.iloc[:,4]
print(sales)

fig=figure()
ax=Axes3D(fig)
ax.scatter(TV,radio,sales)
# ax.set_title('Scatter plot of x-y pairs semi-focused in two regions')
# ax.set_xlabel('x value')
# ax.set_ylabel('y value')
# plt.show()

X=pd.concat([TV,radio,newspaper],axis=1)
Y=sales

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X,Y,test_size=0.25)
reg = linear_model.LinearRegression()

reg.fit(X_train, Y_train)
Y_pred = reg.predict(X_test)

Coef = reg.coef_
R2 = r2_score(Y_test,Y_pred)
MSE = mean_squared_error(Y_test,Y_pred)

print(Coef,R2,MSE)

style.use('ggplot')

plt.scatter(Y_test,Y_pred, color='black')
plt.show()

fig2 = figure()
ax2=Axes3D(fig2)
ax2.scatter(X_test['TV'],X_test['Radio'],Y_test,color='blue')
ax2.scatter(X_test['TV'],X_test['Radio'],Y_pred,color='green')
plt.title('Real Data vs Predicted Data')
plt.xlabel('TV')
plt.ylabel('Radio')
plt.show()