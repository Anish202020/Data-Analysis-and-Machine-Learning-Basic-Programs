import numpy as np
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

X,Y = make_moons(n_samples= 200,shuffle = True,noise = 0.5, random_state = None)
# print(X.shape , Y.shape)
# print(X,Y) 

plt.scatter(X[:,0],X[:,1],c=Y,marker='^')
plt.show()

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3)

model = MLPClassifier(hidden_layer_sizes=(5,), activation='logistic',solver='adam',max_iter=5000,verbose=10,tol=0.00000001)

model.fit(X_train, Y_train)

predict = model.predict(X_test)
print(predict)

print(model.score(X_test,Y_test))

Test = plt.scatter(X_test[:,0],Y_test)
Predict = plt.scatter(X_test[:,0],predict)

plt.legend([Predict,Test],["Predicted Data","Real Data"])

plt.show()
