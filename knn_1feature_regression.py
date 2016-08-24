from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
import numpy as np


boston = datasets.load_boston()
X,y = boston.data,boston.target
names = boston["feature_names"]

lr = LinearRegression()
#rank all features, i.e continue the elimination until the last one
rfe = RFE(lr, n_features_to_select=1)
rfe.fit(X,y)
 
print ("Features sorted by their rank:")
print (sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), names)))

X = boston.data[:, 4:5]
y = boston.target

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)



knn = KNeighborsRegressor(n_neighbors=10)

knn.fit(X_train,y_train)

T = np.linspace(0.3, 0.7, 500)[:, np.newaxis]

plt.scatter(X_train, y_train, color = 'black',label = 'data')
plt.plot(T, knn.predict(T), color = 'blue', linewidth = 3,label = 'prediction')
plt.axis('tight')
plt.legend()

plt.show()



