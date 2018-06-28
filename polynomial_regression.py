# -*- coding: utf-8 -*-
import csv
import time
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from numba import jit

X = []
y = []

def get_data(filename):
    with open(filename, 'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader) # skipping column names
        for row in csvFileReader:
            date_str = str(row[0])
            pattern = '%Y-%m-%d'
            epoch = int(time.mktime(time.strptime(date_str, pattern)))
            X.append(epoch)
            y.append(float(row[4]))

    return


get_data('SENSEX2.csv')
X = np.reshape(X, (len(X), 1)) # converting to matrix of n X 1
y = np.reshape(y, (len(y), 1))

# Splitting data into test and train sets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/6, random_state = 0)

# Testing a fit in Linear Model
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)


# Fitting in Polynomial Features
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 20)
X_poly_train = poly_reg.fit_transform(X_train)
X_poly_test = poly_reg.fit_transform(X_test)
poly_reg.fit(X_poly_train,y_train)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly_train,y_train)

# Predict y for polynomial features
y_pred_poly = lin_reg_2.predict(X_poly_test)


# Checking the accuracy
score = lin_reg_2.score(X_poly_test,y_test)

# Visualising the Linear Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Stock Predictior')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.show()


# Visualising the Polynomial Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Stock Predictior (Polynomial Regression)')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.show()


# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Stock Predictior (Polynomial Regression (Smoother))')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()


# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_poly_test, y_pred_poly, color = 'blue')
plt.title('Date vs Price (Test set)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()