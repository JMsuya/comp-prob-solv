import numpy as np

def initialize_rectangle_lattice(rows,columns):
    '''
    It's just np.zeros with different wording
    '''
    return np.zeros((rows,columns))


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


def compute_interaction_energy(site, particle, lattice, neighbor_indices, epsilon_AA=0., epsilon_BB=0., epsilon_AB=0.):
    energy = 0.
    for neighbor in neighbor_indices[tuple(site)]:
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


def attempt_change(lattice, N_A, N_B, N_0, neighbor_indices, param):
    beta = 1/param["T"]
    if (np.random.randint(2) == 1): # Addition
        if (N_0 == 0):
            return lattice, N_A, N_B, N_0
        zeros = np.transpose(np.nonzero(lattice==0))
        site = tuple(zeros[np.random.randint(zeros.shape[0])])
        
        if (np.random.randint(2) == 0): # Particle 1
            particle = 1
            mu = param["mu_A"]
            epsilon = param["epsilon_A"]
            N = N_A
        else: # Particle 2
            particle = 2
            mu = param["mu_B"]
            epsilon = param["epsilon_B"]
            N = N_B
        delta_E = epsilon * compute_interaction_energy(site, particle, lattice, neighbor_indices, param["epsilon_AA"], param["epsilon_BB"], param["epsilon_AB"])
        prob = np.min([1, N_0 / (N + 1) * np.exp(-beta * (delta_E - mu))])
        if (np.random.rand() < prob):
            if (particle == 1):
                N_A += 1
            else:
                N_B += 1
            lattice[site] = particle
            N_0 -= 1
    else: # Removing
        if (N_0 == lattice.size):
            return lattice, N_A, N_B, N_0
        occupied = np.transpose(np.nonzero(lattice))
        site = tuple(occupied[np.random.randint(occupied.shape[0])])
        particle = lattice[site]
        if (particle == 1):
            mu = param["mu_A"]
            epsilon = param["epsilon_A"]
            N = N_A
        else:
            mu = param["mu_B"]
            epsilon = param["epsilon_B"]
            N = N_B
        delta_E = -epsilon * -compute_interaction_energy(site, particle, lattice, neighbor_indices, param["epsilon_AA"], param["epsilon_BB"], param["epsilon_AB"])
        prob = np.min([1, N / (N_0 + 1) * np.exp(-beta * (delta_E + mu))])
        if (np.random.rand() < prob):
            if (particle == 1):
                N_A -= 1
            else:
                N_B -= 1
            lattice[site] = 0
            N_0 += 1
    return lattice, N_A, N_B, N_0


def run_simulation(rows, columns, cycles, params, seed = 30):
    np.random.seed(seed)
    lattice = initialize_rectangle_lattice(rows,columns)
    neighbors = compute_neighbors(lattice)
    N_sites = lattice.size
    N_A = 0
    N_B = 0
    N_0 = N_sites
    coverage_A = np.zeros(cycles)
    coverage_B = np.zeros(cycles)
    for cycle in range(cycles):
        lattice, N_A, N_B, N_0 = attempt_change(lattice, N_A, N_B, N_0, neighbors, params)
        coverage_A[cycle] = N_A/N_sites
        coverage_B[cycle] = N_B/N_sites
    return lattice, coverage_A, coverage_B


params = {
            'epsilon_A': -0.1,
            'epsilon_B': -0.1,
            'epsilon_AA': 0,
            'epsilon_BB': 0,
            'epsilon_AB': 0,
            'mu_A': 0,
            'mu_B': 0,
            'T': .01
}

print(run_simulation(6,4,10000,params)[0])