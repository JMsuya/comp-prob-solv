import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.constants import calorie
from scipy.stats import t

kcal = calorie*1000 # Joules per kilocalorie

# OLS functions are from Lecture 7
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

# All following functions from Lecture 8
def sse(residuals):
    return np.sum(residuals ** 2)

def variance(residuals):
    return sse(residuals) / (len(residuals) - 2)

def se_slope(x, residuals):
    # numerator
    numerator = variance(residuals)
    # denominator
    x_mean = np.mean(x)
    denominator = np.sum((x - x_mean) ** 2)
    return np.sqrt(numerator / denominator)

def se_intercept(x, residuals):
    # numerator
    numerator = variance(residuals)
    # denominator
    x_mean = np.mean(x)
    denominator = len(x) * np.sum((x - x_mean) ** 2)
    return np.sqrt(numerator / denominator)

def confidence_interval_slope(x, residuals, confidence_level):
    # Calculate the standard error of the slope
    se = se_slope(x, residuals)

    # Calculate the critical t-value
    n_data_points = len(x)
    df = n_data_points - 2  # degrees of freedom
    alpha = 1 - confidence_level
    critical_t_value = t.ppf(1 - alpha/2, df)

    # Calculate the confidence interval
    return critical_t_value * se

def confidence_interval_intercept(x, residuals, confidence_level):
    # Calculate the standard error of the intercept
    se = se_intercept(x, residuals)

    # Calculate the critical t-value
    n_data_points = len(x)
    df = n_data_points - 2  # degrees of freedom
    alpha = 1 - confidence_level
    critical_t_value = t.ppf(1 - alpha/2, df)

    # Calculate the confidence interval
    return critical_t_value * se

data = pd.read_csv(r"C:\Users\msuya\Downloads\trouton.csv") # Importing the data to a dataframe

# Calculating the OLS slope and intercept for the data
slope, intercept = ols(data["T_B (K)"],data["H_v (kcal/mol)"]*kcal)

# Calculating the 95% confidence intervals for the slope and intercept
line = slope*data["T_B (K)"]+intercept
residuals = (data["H_v (kcal/mol)"]*kcal)-line
err_slope = confidence_interval_slope(data["T_B (K)"],residuals,.95)
err_intercept = confidence_interval_intercept(data["T_B (K)"],residuals,.95)


print(f"The slope is {slope:.0f} +/- {err_slope:.0f} J/(mol*K)")
# The calculated slope is noticibly higher than Trouton's rule of around 88 J/(mol*K) for the entropy of vaporization.
# It's clear that the data provided doesn't align well with Trouton's rule.

print(f"The intercept is {intercept/1000:.3f} +/- {err_intercept/1000:.3f} kJ/mol")

# Creating the directory if it does not exist
if (not("homework-3-1" in os.listdir())):
    os.makedirs(os.getcwd()+r"\homework-3-1")

# Plotting each class in the data as a different color using multiple "for" loops
ms = []
for m in data["Class"]:
    if not(m in ms):
        ms.append(m)
for n in ms:
    i=[]
    j=[]
    for x in data.index:
        if (data.at[x,"Class"]==n):
            i.append(data.at[x,"T_B (K)"])
            j.append(data.at[x,"H_v (kcal/mol)"]*calorie)
    plt.scatter(i,j, label=n)

# plotting the linear regression
vals = np.linspace(0,2500,1001)
plt.plot(vals, (slope*vals+intercept)/1000, color="black", linestyle="--", label=f"$H_v = a*T_B + b$")

# displaying the values of "a" and "b"
plt.text(1200,55,f"a = {slope:.0f} +/- {err_slope:.0f} J/(mol*K)",fontsize=12)
plt.text(1200,35,f"b = {intercept/1000:.3f} +/- {err_intercept/1000:.3f} kJ/mol", fontsize=12)

# Formatting the plot and saving to the created directory
plt.xlabel(r"$Boiling\ Point\ (K)$")
plt.ylabel(r"$Enthalpy\ of\ Vaporization\ (\ kJ/mol\ )$")
plt.title("Trouton's Rule")
plt.grid()
plt.legend()
plt.savefig(os.getcwd()+r"\homework-3-1\Troutons_Rule_Graph.png")