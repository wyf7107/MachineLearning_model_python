from sklearn import datasets

iris = datasets.load_iris()
X = iris.data
y = iris.target

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


#Standardizing the features:
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(5, p=2, metric='minkowski')
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
print('Misclassified samples: %d' %(y_test != y_pred).sum())

from sklearn.metrics import accuracy_score
print('Accuracy: %.2f' % accuracy_score(y_test,y_pred))