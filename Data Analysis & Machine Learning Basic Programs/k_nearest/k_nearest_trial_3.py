import numpy as np

from sklearn import neighbors,datasets,model_selection
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score

import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

from collections import Counter

iris = datasets.load_iris()
X=iris.data[:,[1,3]]
Y=iris.target

X_train,X_test,Y_train,Y_test = model_selection.train_test_split(X,Y,test_size=0.3)

def train(X_train,Y_train):
    return

def predict(X_train,Y_train,X_test,k):
    distances=[]
    targets=[]
    
    for i in range(len(X_train)):
        distance = np.sqrt(np.sum(np.square(X_test-X_train[i,:])))
        distances.append([distance,i])
    
    distances=sorted(distances)
        
    for i in range(k):
        index=distances[i][1]
        
        targets.append(Y_train[index])
        
    return Counter(targets).most_common(1)[0][0]

def KNN(X_train,Y_train,X_test,prediction,k):
    train(X_train,Y_train)
    
    for i in range(len(X_test)):
        prediction.append(predict(X_train,Y_train,X_test[i,:],k))
    return prediction

prediction=[]
k=5
KNN(X_train,Y_train,X_test,prediction,k)
    
prediction=np.asarray(prediction)

acc=accuracy_score(Y_test,prediction)
print(acc)