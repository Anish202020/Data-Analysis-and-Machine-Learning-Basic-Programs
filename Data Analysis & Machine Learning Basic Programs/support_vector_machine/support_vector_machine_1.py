import numpy as np

from sklearn import datasets,model_selection,svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

import matplotlib.pyplot as plt

from mlxtend.plotting import plot_decision_regions

iris = datasets.load_iris()
X=iris.data[:,:2]
Y=iris.target

X_train,X_test,Y_train,Y_test = model_selection.train_test_split(X,Y,test_size=0.2)

svc = svm.SVC(kernel='linear',C=1)
svc.fit(X_train,Y_train)

Y_p = svc.predict(X_test)

acc = accuracy_score(Y_test,Y_p)
print(acc)

# CR = classification_report(Y_test,Y_p)
# CM = confusion_matrix(Y_test,Y_p)
# print(CR , CM)

# svc = svm.SVC(kernel='poly',degree=4,C=1)
# svc.fit(X_train,Y_train)

# Y_p = svc.predict(X_test)

# acc = accuracy_score(Y_test,Y_p)
# print(acc)

# CR = classification_report(Y_test,Y_p)
# CM = confusion_matrix(Y_test,Y_p)
# print(CR , CM)

plot_decision_regions(X_test,Y_p,clf =svc ,legend=2)
plt.title('IRIS Classification by SVM')
plt.show()
