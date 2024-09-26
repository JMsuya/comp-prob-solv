# Length values are in Angstroms, Energy values are in electron volts
import numpy as np
from scipy.optimize import minimize
def lennard_jones(r, epsilon=0.01, sigma=3.4):
    v = 4*epsilon*((sigma/r)**12-(sigma/r)**6)
    return v
print(minimize(lennard_jones,4))
import matplotlib.pyplot as plt
x = np.linspace(3,6,301)
plt.plot(x,lennard_jones(x))
plt.title('Potential energy of Ar dimer vs its bond length')
plt.xlabel('$Distance\ (Angstroms)$')
plt.ylabel('$Potential\ Energy\ (eV)$')
plt.grid(True)
plt.show()