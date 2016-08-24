
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE


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

T = np.linspace(0.3, 0.7, 500)[:, np.newaxis]

rfo = RandomForestRegressor()
rfo.fit(X_train,y_train)

plt.figure()
plt.scatter(X, y, c="k", label="data")
plt.plot(T, rfo.predict(T), c="g", label="max_depth=2", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()