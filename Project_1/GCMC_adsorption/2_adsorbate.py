import numpy as np
import matplotlib.pyplot as plt

def initialize_rectangle_lattice(rows,columns):
    '''
    It's just np.zeros with different wording
    '''
    return np.zeros((rows,columns))


def compute_neighbors(lattice):
    '''
    Generate a dictionary of the indices of adjacent cells in a rectangular lattice.

    The dictionary is organized with keys being tuples representing the index pair of the central point.
    The corresponding value is a tuple of tuples representing the index pairs of the four cardinally adjacent points on the lattice.
    The edges of the lattice act periodically, as in the first cell in a row will have the last cell in the row as a neighbor and vice-versa.
    The same is true for columns.

    Parameters
    ----------
    lattice : 2darray
        A 2-dimensional array representing the cellular lattice
    
    Returns
    -------
    dict[tuple]
        Dictionary of index pairs(row,column) organized as (cell):((left),(right),(up),(down))
    '''
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
    '''
    Calculate the interaction energy between an absorbed particle and its neighbors.

    Calculate the total energy of a particle in a cell on a rectangular lattice interacting with the particles in adjacent cells.
    The energy unit of the output is the same as that of the 'epsilon' inputs.

    Parameters
    ----------
    site : tuple_like
        Position of central particle in lattice.
    particle : int
        Particle occupying 'site'. 1 = A, 2 = B.
    lattice : 2darray
        The lattice containing the particle.
    neighbor_indices : dict[tuple,tuple]
        Dictionary of all point adjacents from compute_neighbors().
    epsilon_AA : float, default: 0
        Interaction energy between two adjacent A particles.
    epsilon_BB : float, default: 0
        Interaction energy between two adjacent B particles.
    epsilon_AB : float, default: 0
        Interaction energy between an A and a B particle.
    
    Returns
    -------
    float
        The total energy of the particle interacting with adjacent particles.
    '''
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
    '''
    Runs a GCMC adsorption simulation on a rectangular lattice.

    Generates a 2d array representation of a rectangular lattice, then uses a Grand Canonical Monte Carlo simulation 
    to model competitive adsorption of two species.

    Parameters
    ----------
    rows : int
        The number of rows of the rectangular lattice
    columns : int
        The number of columns of the rectangular lattice
    cycles : int
        The number of times the adsorption algorithm will be run
    params : dict[str,float]
        A dictionary with keywords 'epsilon_A', 'epsilon_B', 'epsilon_AA', 'epsilon_BB', 'epsilon_AB', 'mu_A', 'mu_B', 'T'
    seed : int, default: 30
        The seed to use with the numpy random number generator
    Returns
    -------
    2darray
        The final lattice generated by the last cycle,
    1darray
        The coverage of adsorbate A indexed by cycle,
    1darray
        The coverage of adsorbate B indexed by cycle
    '''
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


def plot_lattice(lattice, ax, title):
    '''
    Generates a matplotlib.axes object of a rectangular lattice.

    Parameters
    ----------
    lattice : 2darray
        A 2-dimensional array representing the cellular lattice
    ax : axes
        The axis the lattice will be displayed on
    title : str
        A title displayed above the plot
    
    Returns
    -------
    axes
        the original axes object entered into ax
    '''
    ax.axis([0,lattice.shape[1],0,lattice.shape[0]])
    ax.axis("scaled")
    for y in range(lattice.shape[0]):
        for x in range(lattice.shape[1]):
            if (lattice[y][x] == 1):
                ax.scatter(x+.5, y+.5, 250, color="r")
            elif (lattice[y][x] == 2):
                ax.scatter(x+.5, y+.5, 250, color="b")
    ax.tick_params("both", direction="in", labelbottom=False, labelleft=False, color="0.7", grid_color="0.7")
    ax.set_xticks(range(1,lattice.shape[1]))
    ax.set_yticks(range(1,lattice.shape[0]))
    ax.grid(True)
    ax.set_title(title)    
    return ax



params = {
    'epsilon_A': -0.1,
    'epsilon_B': -0.1,
    'epsilon_AA': 0,
    'epsilon_BB': 0,
    'epsilon_AB': 0,
    'mu_A': -0.1,
    'mu_B': -0.1,
    'T': .01  # Temperature (in units of k)
}