import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.constants import R, calorie

kcal = calorie*1000 # Joules per kilocalorie

def ols_slope(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    return numerator / denominator

def ols_intercept(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    slope = ols_slope(x, y)
    return y_mean - slope * x_mean

def ols(x, y):
    slope = ols_slope(x, y)
    intercept = ols_intercept(x, y)
    return slope, intercept

data = pd.read_csv(r"C:\Users\msuya\Downloads\trouton.csv")
slope, intercept = ols(data["T_B (K)"],data["H_v (kcal/mol)"]*kcal)
print(f"The slope is {slope:.1f} j/(mol*K)")
# The calculated slope is noticibly higher than Trouton's rule of around 88 J/(mol*K) for the entropy of vaporization.
# It's clear that the data provided doesn't align well with Trouton's rule.

plt.scatter(data["T_B (K)"],data["H_v (kcal/mol)"]*kcal)
plt.plot(data["T_B (K)"], slope*data["T_B (K)"]+intercept)

plt.show()