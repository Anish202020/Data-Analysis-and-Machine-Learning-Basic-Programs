import numpy as np

from sklearn import datasets,model_selection,svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.utils import shuffle

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from mlxtend.data import mnist_data
from mlxtend.plotting import plot_decision_regions
# HEAT MAP
import seaborn as sns

X,Y=mnist_data()
X,Y=shuffle(X,Y,random_state=0)
X=X[1:1001]
Y=Y[1:1001]
# print(X.shape)

def plot_digit(X,Y,idx):
    img=X[idx].reshape(28,28)
    plt.imshow(img,cmap='Greys')
    plt.title('True Label: %d' %Y[idx])
    plt.show()
    
# plot_digit(X,Y,96)

X_train,X_test,Y_train,Y_test = model_selection.train_test_split(X,Y,test_size=0.2)

# svc = svm.SVC(kernel='rbf',C=10)
# svc.fit(X_train,Y_train)

# Y_p = svc.predict(X_test)
# acc = accuracy_score(Y_test,Y_p)
# print(acc)

# CR = classification_report(Y_test,Y_p)
# CM = confusion_matrix(Y_test,Y_p)
# print(CR,CM)



svc= svm.SVC(kernel='linear',C=10)
svc.fit(X_train,Y_train)

Y_p = svc.predict(X_test)
acc = accuracy_score(Y_test,Y_p)
print(acc)

CR = classification_report(Y_test,Y_p)
CM = confusion_matrix(Y_test,Y_p)
print(CR,CM)

plt.figure(figsize=(9,9))
sns.heatmap(CM,annot=True,cmap='Blues_r')
plt.show()