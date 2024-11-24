import numpy as np
import matplotlib.pyplot as plt

def initialize_rectangle_lattice(rows,columns):
    '''
    It's just np.zeros with different wording
    '''
    return np.zeros((rows,columns),int)


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


def compute_interaction_energy(site, particle, lattice, neighbor_indices, params):
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
    params : 2DArray
    
    Returns
    -------
    float
        The total energy of the particle interacting with adjacent particles.
    '''
    energy = 0.

    for neighbor in neighbor_indices[tuple(site)]:
        if (lattice[neighbor] != 0):
            energy += params[particle][lattice[neighbor]]
    
    return energy


def attempt_change(lattice, counts, neighbor_indices, params):
    
    beta = 1/params[0][0]

    if (np.random.randint(2) == 1): # Addition. 50% chance
        if (counts[0] == 0): # if lattice is full
            return lattice, counts
        
        zeros = np.transpose(np.nonzero(lattice==0))
        site = tuple(zeros[np.random.randint(zeros.shape[0])])
        
        particle = np.random.randint(1,params.shape[0])
        mu = params[0][particle]
        epsilon = params[particle][0]
        N = counts[particle]

        delta_E = epsilon + compute_interaction_energy(site, particle, lattice, neighbor_indices, params)
        prob = np.min([1, counts[0] / (N + 1) * np.exp(-beta * (delta_E - mu))])
        if (np.random.rand() < prob):
            counts[particle] += 1
            lattice[site] = particle
            counts[0] -= 1
    

    else: # Removing. 50% chance
        if (counts[0] == lattice.size): # if lattice is empty
            return lattice, counts
        
        occupied = np.transpose(np.nonzero(lattice))
        site = tuple(occupied[np.random.randint(occupied.shape[0])])

        particle = lattice[site]
        mu = params[0][particle]
        epsilon = params[particle][0]
        N = counts[particle]
        
        delta_E = -epsilon - compute_interaction_energy(site, particle, lattice, neighbor_indices, params)
        prob = np.min([1, N / (counts[0] + 1) * np.exp(-beta * (delta_E + mu))])
        if (np.random.rand() < prob):
            counts[particle] -= 1
            lattice[site] = 0
            counts[0] += 1
    
    return lattice, counts


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
    params : 2DArray
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
    N_sites = float(lattice.size)
    counts = np.zeros(params.shape[0], int)
    counts[0] = N_sites
    coverages = np.zeros((cycles,params.shape[0]), float)
    for cycle in range(cycles):
        lattice, counts = attempt_change(lattice, counts, neighbors, params)
        coverages[cycle] = counts/N_sites
    coverages = coverages.transpose()
    return lattice, coverages

def plot_lattice(lattice, ax, title, colors):
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
            if (lattice[y][x] != 0):
                ax.scatter(x+.5, y+.5, 40, color=colors[lattice[y][x]])

    ax.tick_params("both", direction="in", labelbottom=False, labelleft=False, color="0.7", grid_color="0.7")
    ax.set_xticks(range(1,lattice.shape[1]))
    ax.set_yticks(range(1,lattice.shape[0]))
    ax.grid(True)
    ax.set_title(title)    
    return ax


# parameters template:
#params = np.array([
#    [T,         mu_A,       mu_B,       mu_C,       etc],
#    [epsilon_A, epsilon_AA, epsilon_AB, epsilon_AC, etc],
#    [epsilon_B, epsilon_AB, epsilon_BB, epsilon_BC, etc],
#    [epsilon_C, epsilon_AC, epsilon_BC, epsilon_CC, etc],
#    [etc,       etc,        etc,        etc,        etc]
#])
# etc is if you want more molecules added to the simulation
params =np.array([
    [.02, 0, 0, -.05],
    [-.1, .05, .05, -.2],
    [-.1, .05, .05, -.2],
    [-.1, -.2, -.2, .05]
])
params2 =np.array([
    [.02, 0, 0],
    [-.1, .05, .05],
    [-.1, .05, .05]
])
final, coverage = run_simulation(10,12,10000,params)
colors = ["1", "r", "b", "y"]
final2, coverage2 = run_simulation(10,12,10000,params2, 20)
fig, axs = plt.subplot_mosaic([["A","B"],["C","D"]])
plot_lattice(final, axs["A"],"3rd species",colors)
plot_lattice(final2, axs["B"],"Control",colors)
axs["C"].plot(np.linspace(0,9999,10000), coverage[2])
axs["D"].plot(np.linspace(0,9999,10000), coverage2[2])
axs["C"].set_ylim(0,.4)
axs["D"].set_ylim(0,.4)
plt.show()