# Length values are in Angstroms, Energy values are in electron volts
import numpy as np
from scipy.optimize import minimize
import math
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
