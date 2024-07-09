import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

from keras import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv("temperature.csv")
L = len(df)
print(L)

TAVG = np.array([df.iloc[:,4]])
TMAX = np.array([df.iloc[:,3]])
TMIN = np.array([df.iloc[:,5]])

# print(TAVG,TMAX,TMIN)

fig = plt.figure(1)
ax = fig.add_subplot(111,projection='3d')

ax.scatter(TMAX,TMIN,TAVG,marker='o')
ax.set_xlabel('TMAX')
ax.set_xlabel('TMIN')
ax.set_xlabel('TAVG')
plt.show()

plt.plot(TAVG[0,:])
plt.show()

X = np.concatenate([TMIN,TMAX],axis=0)
X = np.transpose(X)

Y = np.transpose(TAVG)


sc = MinMaxScaler()
sc.fit(X)
X=sc.transform(X)

sc1 = MinMaxScaler()
sc1.fit(Y)
Y=sc1.transform(Y)

X = np.reshape(X,(X.shape[0],1,X.shape[1]))

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3)

model = Sequential()
model.add(LSTM(10,activation='tanh',input_shape=(1,2),recurrent_activation='hard_sigmoid'))

model.add(Dense(1))
model.compile(loss='mean_squared_error' ,optimizer='rmsprop',metrics=[metrics.mae])

model.fit(X_train,Y_train,epochs=50,verbose=2)

predict = model.predict(X_test)

plt.figure(2)
plt.scatter(Y_test,predict)
plt.show(block = False)

plt.figure(3)
Real = plt.plot(Y_test)
Predict = plt.plot(predict)
plt.show()
