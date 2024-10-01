# Length values are in Angstroms, Energy values are in electron volts
import numpy as np
from scipy.optimize import minimize
import math
import os
def lennard_jones(r, epsilon=0.01, sigma=3.4):
    v = 4*epsilon*((sigma/r)**12-(sigma/r)**6)
    return v
def compute_bond_length(atom1:list,atom2:list): # redefining all the code from homework-1-2
    '''
    Computes the distance between two atoms
    Parameters:
    atom1 (list): Coordinates of the first atom
    atom2 (list): Coordinates of the second atom
    Returns:
    float: distance between the two atoms
    '''
    return ((atom1[0]-atom2[0])**2+(atom1[1]-atom2[1])**2+(atom1[2]-atom2[2])**2)**.5
Ar1 = [0.000,0.000,0.000]
Ar2 = [0.000,0.000,round(float(minimize(lennard_jones,4).x[0]),3)]
print("Optimal distance:",compute_bond_length(Ar1,Ar2))

if (not("homework-2-1" in os.listdir())): # Creating the directory if it does not exist
    os.makedirs(os.getcwd()+r"\homework-2-1")

f = open(os.getcwd()+r"\homework-2-1\Ar2.xyz","w",encoding="utf-8") # encoding the file
f.write("2\nArgon dimer molecule\nAr ") # Writing the .xyz file
for x in Ar1:
    f.write(str(x)+" ")
f.write("\nAr ")
for y in Ar2:
    f.write(str(y)+" ")
f.close()
print()
print("Generated .xyz File:")
g = open(os.getcwd()+r"\homework-2-1\Ar2.xyz","r") # Rechecking that the file wrote successfully
for line in g:
    print(line)