from sklearn import *
from sklearn import datasets,linear_model,model_selection
from sklearn.metrics import mean_squared_error, r2_score,accuracy_score
from matplotlib.pyplot import *
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

boston = datasets.load_boston()
# print(boston.keys())
# print(boston.DESCR)
# print(boston.data.shape)

# X = boston.data( : , 1:13)
X = boston.data[:,1:13]
Y = boston.target

X_train ,X_test,Y_train,Y_test = model_selection.train_test_split(X,Y,test_size=0.25)
reg = linear_model.LinearRegression()

reg.fit(X_train,Y_train)

Y_pred = reg.predict(X_test)

Coef = reg.coef_
print(Coef)

R2 = r2_score(Y_test,Y_pred)
MSE = mean_squared_error(Y_test,Y_pred)
print(R2,MSE)

style.use('ggplot')
plt.scatter(Y_pred,Y_test,color='black')
plt.title('Predicted Values vs Real Values')
plt.xlabel('Y_pred')
plt.ylabel('Y_test')
plt.show()