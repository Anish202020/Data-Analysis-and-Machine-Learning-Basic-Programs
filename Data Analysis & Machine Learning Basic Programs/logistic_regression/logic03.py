from sklearn import *
from sklearn import datasets,model_selection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score,mean_squared_error,classification_report,confusion_matrix
from sklearn.utils import shuffle

from sklearn.datasets._samples_generator import make_blobs

import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import seaborn as sns

from mlxtend.plotting import plot_decision_regions
from mlxtend.data import mnist_data

X,Y = mnist_data()
X,Y=shuffle(X,Y,random_state=0)

X=X[1:1001]
Y = Y[1:1001]
print(X.shape)
print(Y.shape)

def plot_digits(X,Y,idx):
    img=X[idx].reshape(28,28)
    plt.imshow(img,cmap='Greys')
    plt.title('true label: %d' %Y[idx])
    plt.show()
    
plot_digits(X,Y,96)

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X,Y,test_size=0.25)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

logreg = linear_model.LogisticRegression(C=1e5,random_state=0)

logreg.fit(X_train_std,Y_train)

Y_pred = logreg.predict(X_test_std)

C_M = confusion_matrix(Y_test,Y_pred)
C_R = classification_report(Y_test,Y_pred)
R2 = r2_score(Y_test,Y_pred)
MSE = mean_squared_error(Y_test,Y_pred)

print(C_M,C_R,R2,MSE)

plt.figure(figsize=(10,10))
sns.heatmap(C_M,annot=True,cmap='Blues_r')
plt.xlabel('Predicted Values')
plt.ylabel('Real Values')

plt.show()