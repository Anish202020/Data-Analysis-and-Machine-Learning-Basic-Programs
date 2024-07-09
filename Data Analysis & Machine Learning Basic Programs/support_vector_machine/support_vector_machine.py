import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm

x=[2.0,1.5,8.0,7.0,9.0,2.5,0.8]
y=[1.1,2.0,7.0,7.5,8.0,2.7,1.2]

# plt.scatter(x,y)
# plt.show()

X = np.array([[2,1.1],[1.5,2],[8,7],[7,7.5],[9,8],[2.5,2.7],[0.8,1.2]])
Y = [0,0,1,1,1,0,0]

clf = svm.SVC(kernel='linear',C=1)

clf.fit(X,Y)

p1 = clf.predict([[0.7,0.8]])
print(p1)

p2 = clf.predict([[8.5,7]])
print(p2)

w = clf.coef_
print(w)

a = -w[0][0]/w[0][1]
print(a)

x=np.linspace(0,11)
y=a*x-clf.intercept_[0]/w[0][1]

plt.plot(x,y,color='blue')
plt.scatter(X[:,0],X[:,1])
plt.legend()
plt.show()

