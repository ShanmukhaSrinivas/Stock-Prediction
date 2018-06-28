# -*- coding: utf-8 -*-
"""import time
date_str = '2010-01-04'
pattern = '%Y-%m-%d'
t = time.strptime(date_str, pattern)
t = t[7] //7 + 1
"""
import csv
import time
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from numba import jit

X = []
y = []


def get_data(filename):
    week = 0
    with open(filename, 'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader) # skipping column names
        for row in csvFileReader:
            date_str = str(row[0])
            pattern = '%Y-%m-%d'
            t = time.strptime(date_str, pattern)
            calc_week = t[7] //7 + 1 + (t[0]%2010)
            if week < calc_week:
                week += 1
            X.append(week)
            y.append(float(row[4]))

    return


get_data('SENSEX.csv')
X = np.reshape(X, (len(X), 1)) # converting to matrix of n X 1
y = np.reshape(y, (len(y), 1))