import numpy as np
from scipy.constants import k,electron_volt

def initialize_rectangle_lattice(rows,columns):
    return np.zeros((rows,columns))

lattice = initialize_rectangle_lattice(5,3)
lattice[3][2] = 1
lattice[1][2] = 2


def compute_neighbors(lattice):
    neighbor_indices = {}
    for row in range(lattice.shape[0]):
        for col in range(lattice[row].shape[0]):
            neighbor_indices[(row,col)] = (
                ((row-1) % lattice.shape[0], col),
                ((row+1) % lattice.shape[0], col),
                (row, (col-1) % lattice[row].shape[0]),
                (row, (col+1) % lattice[row].shape[0])
            )
    return neighbor_indices

neighbors = compute_neighbors(lattice)

def compute_interaction_energy(site, particle, lattice, neighbor_indices, epsilon_AA=0, epsilon_BB=0, epsilon_AB=0):
    energy = 0.
    for neighbor in neighbor_indices[site]:
        if (lattice[neighbor] != 0):
            if (particle == 1):
                if (lattice[neighbor] == 1):
                    energy += epsilon_AA
                if (lattice[neighbor] == 2):
                    energy += epsilon_AB
            if (particle == 2):
                if (lattice[neighbor] == 1):
                    energy += epsilon_AB
                if (lattice[neighbor] == 2):
                    energy += epsilon_BB
    return energy

I_E = compute_interaction_energy((2,2),2,lattice,neighbors,1,2,.5)

def attempt_change(lattice, N_A, N_B, N_0, neighbor_indices, params):
    2