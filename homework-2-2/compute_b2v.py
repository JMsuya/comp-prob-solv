import numpy as np
from scipy.integrate import trapezoid
import math
from scipy.constants import k, eV, pi, N_A
import pandas as pd
import os
import matplotlib.pyplot as plt # Importing everything
k_B = k/eV # Converting k(J/K) to K_B(eV/K)

def hard_sphere(r, sigma=3.4): # Defining each potential function
    if(r<sigma): # Infinity approximated as 1000
        return 1000
    else:
        return 0

def square_well(r, epsilon=.01, sigma=3.4, lam=1.5):
    if (r<sigma):
        return 1000
    elif (r<(sigma*lam)):
        return -(epsilon)
    else:
        return 0

def lennard_jones(r, epsilon=.01, sigma=3.4):
    v = 4*epsilon*(((sigma/r)**12)-(sigma/r)**6)
    return v

def B2v_hard(T,n): # Integrating each function over an approximation of 0 to infinity
    r = np.linspace(.001,(3.4*5),n,endpoint=True)
    u = np.zeros(n) # n is the number of values of r generated. Lowering it would compute faster but be less accurate
    count = 0
    for x in r:
        u[count] = ((math.exp(-(hard_sphere(x)/(k_B*T)))-1)*(x**2))
        count= count+1
    a = -2*pi*N_A*(trapezoid(u,r))
    return a
print("B2V at 100K:")
print("Hard Sphere:",B2v_hard(100,1000))

def B2v_square(T,n):
    v = np.linspace(.001,(3.4*5),n,endpoint=True)
    w = np.zeros(n)
    count = 0
    for x in v:
        w[count] = ((math.exp(-(square_well(x)/(k_B*T)))-1)*(x**2))
        count= count+1
    a = -2*pi*N_A*(trapezoid(w,v))
    return a
print("Square Well:",B2v_square(100,1000))

def B2v_lennard(T,n):
    i = np.linspace(.001,(3.4*5),n,endpoint=True)
    j = np.zeros(n)
    count = 0
    for x in i:
        j[count] = ((math.exp(-(lennard_jones(x)/(k_B*T)))-1)*(x**2))
        count= count+1
    a = -2*pi*N_A*(trapezoid(j,i))
    return a
print("Lennard Jones:",B2v_lennard(100,1000))

if (not("homework-2-2" in os.listdir())): # Creating the directory if it does not exist
    os.makedirs(os.getcwd()+r"\homework-2-2")

T = np.linspace(100,800,701) # X-value of the graph
THard = np.zeros(701)
count2 = 0
while (count2<701):
    THard[count2] = B2v_hard(T[count2],1000)
    count2 = count2+1
plt.plot(T,THard, label=r"$Hard\ Sphere$") # Graphing each function vs temperature

TSquare = np.zeros(701)
count2 = 0
while (count2<701):
    TSquare[count2] = B2v_square(T[count2],1000)
    count2 = count2+1
plt.plot(T,TSquare, label=r"$Square\ Well$")

TLennard = np.zeros(701)
count2 = 0
while (count2<701):
    TLennard[count2] = B2v_lennard(T[count2],1000)
    count2 = count2+1
plt.plot(T,TLennard, label=r"$Lennard\ Jones$")

plt.plot(np.array([100,800]),np.array([0,0]), linestyle="--", color="black", label="$B2V=0$")
plt.xlabel(r"$Temperature\ (K)$")
plt.ylabel(r"$B2V\ (Å^3/mol)$") # Formatting the graph
plt.title("The Second Virial Coefficient of various Potential Equations vs Temperature")
plt.legend()
plt.savefig(os.getcwd()+r"\homework-2-2\B2v-T_graph.png") # Saving the graph

DHS = {"T (K)":T,"B2V (A^3/mol)":THard} # Converting the arrays to dict as converting dict to dataFrame is easier
HS = pd.DataFrame(DHS)
HS.to_csv(os.getcwd()+r"\homework-2-2\B2V_Hard.csv") # Exporting the data to .csv files

DSW = {"T (K)":T,"B2V (A^3/mol)":TSquare}
SW = pd.DataFrame(DSW)
SW.to_csv(os.getcwd()+r"\homework-2-2\B2V_Square.csv")

DLJ = {"T (K)":T,"B2V (A^3/mol)":TLennard}
LJ = pd.DataFrame(DLJ)
LJ.to_csv(os.getcwd()+r"\homework-2-2\B2V_Lennard.csv")

f = open(os.getcwd()+r"\homework-2-2\discussion.md","w",encoding="utf-8")
f.write("# Discussion of B~2V~ from various potential equations\n") # Writing 'discussion' markdown file
f.write("B~2V~ under the *Hard Sphere* model is always positive. Since the minimum energy is 0, this makes sense.\n")
f.write("For the *Square Well* and *Lennard Jones* models, B~2V~ is negative at low-to-room temperatures, modeling that the attractive force between two argon atoms is present and dependent on temperature.\n")
f.write("Just the presence of negative values in the two latter models changes the graph from linear to logrithmic. Since the *Hard Sphere* has no negative potential values, its graph is never negative.")
f.close()
g = open(os.getcwd()+r"\homework-2-2\discussion.md","r")
for line in g:
    print(line)