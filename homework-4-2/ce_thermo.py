import numpy as np
from scipy.constants import k, eV
import matplotlib.pyplot as plt
import os
import pandas as pd

# Converting scipy.constants.k, in J/K, to k_B in eV/K
k_B = k/eV

if (__name__=="__main__"):
    T = np.linspace(300,2000,1701)
    # Arrays are organized as [g_i, E_i]
    ISO = np.array([[14, 0]])
    SOC = np.array([[6, 0],[8, .28]])
    SOC_CFS = np.array([[4, 0],[2, .12],[2, .25],[4, .32],[2, .46]])

    systems = [[ISO, "Isolated"],[SOC, "Spin-Orbit Coupling"],[SOC_CFS, "SOC and Crystal Field Splitting"]]

# Calculates partition function
def partition(states,T):
    Z = 0
    for i in states:
        Z = Z + i[0]*np.exp(-i[1]/(k_B*T))
    return Z

# Calculates all three thermodynamic properties at once and returns a tuple
def thermo_properties(states,T):
    Z = partition(states,T)
    U = -np.gradient(np.log(Z), 1/(k_B*T))
    F = -k_B*T*np.log(Z)
    S = -np.gradient(F,T)
    return U,F,S

if (__name__=="__main__"):
    # Creating a 3-graph figure
    fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(20,5))

    # For each system: plot its thermodynamic properties on their respective graphs and output a .csv with all data used
    for sys in systems:
        ax1.plot(T,thermo_properties(sys[0],T)[0],label=sys[1])
        ax2.plot(T,thermo_properties(sys[0],T)[1],label=sys[1])
        ax3.plot(T,thermo_properties(sys[0],T)[2],label=sys[1])

        data = pd.DataFrame(np.transpose([T, thermo_properties(sys[0],T)[0], thermo_properties(sys[0],T)[1], thermo_properties(sys[0],T)[2]]), columns=["T (K)", "U (eV)", "F (eV)", "S (eV/K)"])
        data.to_csv(sys[1]+".csv")

    # Formatting
    ax1.legend()
    ax1.set_xlabel("Temperature (K)")
    ax1.set_ylabel("Internal Energy (eV)")
    ax1.grid()

    ax2.set_xlabel("Temperature (K)")
    ax2.set_ylabel("Free Energy (eV)")
    ax2.grid()

    ax3.set_xlabel("Temperature (K)")
    ax3.set_ylabel("Entropy (eV/K)")
    ax3.grid()

    fig.suptitle("Thermodynamic Properties of the 4f Electron of $Ce^3$$^+$", fontsize=16)
    plt.tight_layout()
    plt.show()
