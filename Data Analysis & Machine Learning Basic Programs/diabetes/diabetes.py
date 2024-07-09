from sklearn import datasets, model_selection
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score , classification_report,confusion_matrix
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

df = pd.read_csv('diabetes.csv')

X1 = df.iloc[:,2]

X2 = df.iloc[:,3]

X3 = df.iloc[:,6]

Y = df.iloc[:,8]

X = pd.concat([X1,X2,X3],axis=1,)

X_train,X_test,Y_train,Y_test, = model_selection.train_test_split(X,Y,test_size=0.2)

gnb=GaussianNB()
gnb.fit(X_train,Y_train)

Y_p = gnb.predict(X_test)

acc = accuracy_score(Y_test,Y_p)
CM = confusion_matrix(Y_test,Y_p)
CR = classification_report(Y_test,Y_p)
print(acc)
print(CM,CR)

