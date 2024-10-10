import numpy as np
from scipy.integrate import trapezoid
from scipy.constants import R
import pandas as pd
import os

# Defining constants
V_i = .1 # m^3
V_f = 3*V_i # m^3
n = 1 # mol
T = 300 # K
gamma = 1.4

# Defining adiabatic pressure function
def pressure_adi(V,V_0, gamma, n, T_0):
    P = (n*R*T_0*V_0**(gamma-1))/(V**gamma)
    return P

# Defining work function
def work_adi(V, V_0, gamma, n, T_0):
    volume = np.linspace(V_0,V,1001)
    w = -trapezoid(pressure_adi(volume, V_0, gamma, n, T_0),volume)
    return w

# Creating directory if it isn't present
if (not("homework-4-1" in os.listdir())):
    os.makedirs(os.getcwd()+r"\homework-4-1")

# Exporting data to csv using Pandas
data = pd.DataFrame(np.transpose([np.linspace(V_i,V_f,1001),work_adi(np.linspace(V_i,V_f,1001),V_i,gamma,n,T)]),columns=["V_f (m^3)","work (J)"])
data.to_csv(os.getcwd()+r"\homework-4-1\adi_expansion_300K.csv")