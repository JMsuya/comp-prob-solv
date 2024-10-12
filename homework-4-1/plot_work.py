import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Importing the csv's from the other code blocks
iso = pd.read_csv(os.getcwd()+r"\homework-4-1\iso_expansion_300K.csv")

adi = pd.read_csv(os.getcwd()+r"\homework-4-1\adi_expansion_300K.csv")

# Formatting and showing the graph
plt.plot(iso["V_f (m^3)"],iso["work (J)"],label="Isothermal")
plt.plot(adi["V_f (m^3)"],adi["work (J)"],label="Adiabatic")
plt.legend()
plt.title("Expansion of 1 mol of gas at 300 K")
plt.grid()
plt.xlabel(r"$Final\ Volume\ (m^3)$")
plt.ylabel("Work (J)")
plt.show()