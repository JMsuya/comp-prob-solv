# Length values are in Angstroms, Energy values are in electron volts
import numpy as np
import os
from scipy.optimize import minimize
import math
def lennard_jones(r, epsilon=0.01, sigma=3.4):
    v = 4*epsilon*((sigma/r)**12-(sigma/r)**6)
    return v
def compute_bond_length(atom1:list,atom2:list):
    '''
    Computes the distance between two atoms
    Parameters:
    atom1 (list): Coordinates of the first atom
    atom2 (list): Coordinates of the second atom
    Returns:
    float: distance between the two atoms
    '''
    return ((atom1[0]-atom2[0])**2+(atom1[1]-atom2[1])**2+(atom1[2]-atom2[2])**2)**.5
def compute_bond_angle(a:list,b:list,c:list):
    '''
    Computes the bond angle between three atoms
    Parameters:
    a (list): Coordinates of the first outer atom
    b (list): Coordinates of the inner atom
    c (list): Coordinates of the second outer atom
    Returns:
    float: Angle (degrees) between the two bonds
    '''
    d = math.degrees(math.acos(((a[0]-b[0])*(c[0]-b[0])+(a[1]-b[1])*(c[1]-b[1])+(a[2]-b[2])*(c[2]-b[2]))/(compute_bond_length(a,b)*compute_bond_length(b,c))))
    return d
def V(D): # Defining the function to be minimized
    r12 = D[0]
    x3 = D[1]
    y3 = D[2]
    E = lennard_jones(r12)+lennard_jones((x3**2+y3**2)**0.5)+lennard_jones(((r12-x3)**2+y3**2)**0.5)
    return E
values = minimize(V,[4,4,4]).x # calling x gives the optimized values.
Ar1 = [0.0,0.0,0.0]
Ar2 = [round(values[0],3),0.0,0.0]
Ar3 = [round(values[1],3),round(values[2],3),0.0]
print("r12:",round(compute_bond_length(Ar1,Ar2),3))
print("r23:",round(compute_bond_length(Ar2,Ar3),3))
print("r13:",round(compute_bond_length(Ar1,Ar3),3))
print("Ar1 bond angle:",round(compute_bond_angle(Ar2,Ar1,Ar3)))
print("Ar2 bond angle:",round(compute_bond_angle(Ar3,Ar2,Ar1)))
print("Ar3 bond angle:",round(compute_bond_angle(Ar1,Ar3,Ar2)))

if (not("homework-2-1" in os.listdir())): # Generating the directory if it does not exist
    os.makedirs(os.getcwd()+r"\homework-2-1")

f = open(os.getcwd()+r"\homework-2-1\Ar3.xyz","w",encoding="utf-8") # Encoding the file
f.write("3\nArgon trimer molecule\nAr ") # Writing the .xyz file
for x in Ar1:
    f.write(str(x)+" ")
f.write("\nAr ")
for y in Ar2:
    f.write(str(y)+" ")
f.write("\nAr ")
for z in Ar3:
    f.write(str(z)+" ")
f.close()
print()
print("Generated .xyz File:")
g = open(os.getcwd()+r"\homework-2-1\ar3.xyz","r") # Rechecking that the file wrote successfully
for line in g:
    print(line)