import numpy as np
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

X=np.array(
    [[1,4.5],
     [1.2,4.2],
     [1,4],
     [1.5,4.6],
     [5,12],
     [6,11.5],
     [7.5,13],
     [1.5,4.6],
     [8,12],
     [9,13.7]   
    ])
# 0 = cars & 1 = Buses
Y=np.array([0,0,0,0,1,1,1,0,1,1])
 
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')

Weight = X[:,0]
Length = X[:,1]
Type = Y

ax.scatter(Weight,Length,Type,c=Type,marker='^')
ax.set_xlabel('Weight')
ax.set_ylabel('Length')
ax.set_zlabel('Buses or Cars')
plt.show()

model = MLPClassifier(hidden_layer_sizes=(10,),activation='logistic',solver='adam',max_iter=500,verbose=10)

model.fit(X,Y) 
predict = model.predict(X)
print(Y)
print(predict)
print(model.score(X,Y))

Guess=[1.1,4]
print(model.predict([Guess]))