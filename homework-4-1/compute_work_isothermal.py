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

# Defining pressure function
def pressure_iso(V,n,T):
    P = n*R*T/V
    return P

# Defining work function
def work_iso(V_i,V_f,n,T):
    volume = np.linspace(V_i,V_f,1001)
    w = -trapezoid(pressure_iso(volume,n,T),volume)
    return w

# Creating directory if it isn't present
if (not("homework-4-1" in os.listdir())):
    os.makedirs(os.getcwd()+r"\homework-4-1")

# Exporting data to csv using Pandas
data = pd.DataFrame(np.transpose([np.linspace(V_i,V_f,1001),work_iso(V_i,np.linspace(V_i,V_f,1001),n,T)]),columns=["V_f (m^3)","work (J)"])
data.to_csv(os.getcwd()+r"\homework-4-1\iso_expansion_300K.csv")