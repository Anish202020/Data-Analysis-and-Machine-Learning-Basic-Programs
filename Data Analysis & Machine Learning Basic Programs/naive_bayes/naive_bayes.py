from sklearn import datasets, model_selection
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from mlxtend.plotting import plot_decision_regions

iris = datasets.load_iris()
X=iris.data[:,[1,3]]
Y=iris.target

X_train,X_test,Y_train,Y_test =model_selection.train_test_split(X,Y,test_size=0.25)

gnb = GaussianNB()
gnb.fit(X_train,Y_train)

Y_pred = gnb.predict(X_test)

acc = accuracy_score(Y_test,Y_pred)
print(acc)

CM = confusion_matrix(Y_test,Y_pred)
CR = classification_report(Y_test,Y_pred)
print(CM,CR)

plot_decision_regions(X_test,Y_pred,clf=gnb,legend=2)
plt.title("3 class iris classification by using naive bayes classifier")
plt.show()