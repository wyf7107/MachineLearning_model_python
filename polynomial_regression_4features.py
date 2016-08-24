import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
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

X = boston.data[:, 2:6]
y = boston.target



from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

for degree in [2, 3, 4, 5]:
    model = Pipeline([('poly', PolynomialFeatures(degree=degree)),('linear', LinearRegression(fit_intercept=False))])
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    print('MSE train: %.3f, test:%.3f' % (mean_squared_error(y_train,y_train_pred), mean_squared_error(y_test,y_test_pred)))
    print('R^2 train: %.3f, test:%.3f' % (r2_score(y_train,y_train_pred), r2_score(y_test,y_test_pred)))

