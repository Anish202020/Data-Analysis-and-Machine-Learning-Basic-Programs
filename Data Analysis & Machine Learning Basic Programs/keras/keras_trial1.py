import numpy as np

import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense

from keras import metrics

np.random.seed(0)
X = np.random.random((1000,100))
print(X.shape)

Y = np.random.random((1000,1))

model = Sequential()

model.add(Dense(32,activation='relu',input_dim = 100))

model.add(Dense(1,activation='sigmoid'))

model.compile(loss='mean_squared_error',optimizer='rmsprop',metrics=[metrics.mae])

model.fit(X,Y,epochs=1000,batch_size=32,verbose= 2)

Predict = model.predict(X,verbose=1)

plt.figure(1)
plt.scatter(Y,Predict)
plt.show(block=False)

plt.figure(2)
Test = plt.scatter(X[:,0],Y)
Predict = plt.scatter(X[:,0],Predict)
plt.legend([Predict,Test],["Predicted Data","Real Data"])
plt.show()