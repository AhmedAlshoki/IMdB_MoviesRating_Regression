from Preprocessing import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
import timeit
from sklearn.preprocessing import PolynomialFeatures

X_train, X_test, y_train, y_test = Preprocess()


def apply_simple_linear_regression():
    start = timeit.default_timer()

    global X_train, X_test, y_train, y_test
    x_train_selected = X_train['Runtime']
    x_test_selected = X_test['Runtime']

    cls = linear_model.LinearRegression()
    x_train_selected = np.expand_dims(x_train_selected, axis=1)
    y_train = np.expand_dims(y_train, axis=1)
    cls.fit(x_train_selected, y_train)               #Fit method is used for fitting your training data into the model

    x_test_selected = np.expand_dims(x_test_selected, axis=1)
    y_test = np.expand_dims(y_test, axis=1)

    prediction = cls.predict(x_test_selected)

    print('Mean Square Error', metrics.mean_squared_error(y_test, prediction))

    stop = timeit.default_timer()
    print('Time: ', stop - start)

    plt.scatter(x_test_selected, y_test)
    plt.xlabel('Runtime', fontsize=20)
    plt.ylabel('IMDB', fontsize=20)
    plt.plot(x_test_selected, prediction, color='red', linewidth=3)
    plt.show()


def apply_multiple_linear_regression():
    start = timeit.default_timer()

    cls = linear_model.LinearRegression()
    cls.fit(X_train, y_train)
    prediction = cls.predict(X_test)

    print('Mean Square Error', metrics.mean_squared_error(np.asarray(y_test), prediction))

    stop = timeit.default_timer()
    print('Time: ', stop - start)


def apply_polynomial_regression():
    start = timeit.default_timer()

    poly_features = PolynomialFeatures(degree=3)

    # transforms the existing features to higher degree features.
    X_train_poly = poly_features.fit_transform(X_train)

    # fit the transformed features to Linear Regression
    poly_model = linear_model.LinearRegression()
    poly_model.fit(X_train_poly, y_train)

    # predicting on test data-set
    prediction = poly_model.predict(poly_features.fit_transform(X_test))

    print('Mean Square Error', metrics.mean_squared_error(np.asarray(y_test), prediction))

    stop = timeit.default_timer()
    print('Time: ', stop - start)


print("////////////////// Simple Linear Regression: ///////////////////////")
apply_simple_linear_regression()

print("////////////////// Multiple Linear Regression: ///////////////////////")
apply_multiple_linear_regression()

print("////////////////// Polynomial Regression: ///////////////////////")
apply_polynomial_regression()

