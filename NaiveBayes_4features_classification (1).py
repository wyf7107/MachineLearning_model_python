

from IPython.display import Image


from sklearn import datasets
import numpy as np

iris = datasets.load_iris()
X = iris.data[:, :4]
y = iris.target

print('Class labels:', np.unique(y))


#Splitting data into 70% training and 30% test data:
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


#Standardizing the features:
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


                    
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(X_train_std,y_train)

y_pred = gnb.predict(X_test_std)
print('Misclassified samples: %d' %(y_test != y_pred).sum())

from sklearn.metrics import accuracy_score
print('Accuracy: %.2f' % accuracy_score(y_test,y_pred))

