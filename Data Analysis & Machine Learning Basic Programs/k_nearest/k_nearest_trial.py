from sklearn import neighbors,datasets,model_selection
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

X,Y = make_classification(n_features=2,n_redundant=0,n_informative=2,random_state=1,n_clusters_per_class=1)

X_train,X_test,Y_train,Y_test = model_selection.train_test_split(X,Y,test_size=0.5)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

clf = neighbors.KNeighborsClassifier(10,weights='uniform')

clf.fit(X_train_std,Y_train)

Y_pred = clf.predict(X_test_std)

C_M = confusion_matrix(Y_test,Y_pred)
print(C_M)

C_R=classification_report(Y_test,Y_pred)
print(C_R)

plot_decision_regions(X_test_std,Y_pred,clf=clf,legend=2)
plt.title('2 class classification using K nearest Neighbors k=10')
plt.show()
