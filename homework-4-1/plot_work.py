import numpy as np
from scipy.integrate import trapezoid
from scipy.constants import R
import matplotlib.pyplot as plt

V_i = .1 # m^3
V_f = 3*V_i # m^3
n = 1 # mol
T = 300 # K

def pressure(V,n,T):
    P = n*R*T/V
    return P

def work(V_i,V_f,n,T):
    volume = np.linspace(V_i,V_f,1001)
    w = -trapezoid(pressure(volume,n,T),volume)
    return w

volumes = np.linspace(V_i,V_f,1001)
plt.plot(volumes,work(V_i,volumes,n,T))
plt.grid()
plt.xlabel(r"$Final\ Volume\ (m^3)$")
plt.ylabel("Work (J)")
plt.show()