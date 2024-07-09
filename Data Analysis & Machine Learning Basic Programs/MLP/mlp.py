import numpy as np
from sklearn.neural_network import MLPClassifier

X=np.array([[0,0],[0,1],[1,0],[1,1]])

Y=np.array([0,0,0,1])

model = MLPClassifier(hidden_layer_sizes=(50,),activation='logistic',solver='adam',max_iter=1000,verbose=10)

model.fit(X,Y)
predict = model.predict(X)
print(predict)
print(model.score(X,Y))
