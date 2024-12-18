import numpy as np
from scipy.stats import maxwell
import matplotlib.pyplot as plt
import ase
from ase.visualize import view
# Temperature in terms of k_B

def boltzmann_distribution(v, mass, T):
    return np.pi * (mass / (2 * np.pi * T))**(3/2) * v**2 * np.exp(-mass * v**2 / (2 * T))

def minimum_image(delta, box_size):
    displacement = delta - (box_size * np.round(delta / box_size))
    return displacement


def initialize_chain(N_particles, box_size, r0):
    positions = np.zeros((N_particles,3))
    current = [box_size/2,box_size/2,box_size/2]
    positions[0] = current
    i = 1
    while (i<N_particles): #while instead of for to retry in case of overlap
        theta = np.pi*np.random.random()
        phi = 2*np.pi*np.random.random()
        direction = np.array((np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)))
        next = current + r0*direction
        overlap = False
        for j in range(0, i-1): #checking for overlap
            if (np.linalg.norm(next-positions[j]) < r0):
                overlap = True
                break
        if overlap:
            continue
        next = apply_pbc(next, box_size)
        positions[i] = next
        current = next
        i += 1
    return positions

def initialize_velocities(N_particles, T0, mass):
    velocities = np.zeros((N_particles,3))
    for i in range(N_particles):
        theta = np.pi*np.random.random()
        phi = 2*np.pi*np.random.random()
        direction = np.array((np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)))
        velocities[i] = maxwell.rvs() * (3*T0 / mass) * direction
    velocities -= np.mean(velocities,0)
    return velocities

def apply_pbc(position, box_size):
    return (position % box_size)

def compute_harmonic_forces(positions, ks, r0, box_size):
    forces = np.zeros_like(positions)
    for i in range(forces.shape[0]-1):
        displacement = positions[i+1]-positions[i]
        displacement = minimum_image(displacement, box_size)
        distance = np.linalg.norm(displacement)
        magnitude = -ks * (distance - r0)
        force = magnitude * (displacement / distance)
        forces[i] -= force
        forces[i+1] += force
    return forces

def compute_lennard_jones_forces(positions, epsilon_r, epsilon_a, sigma, box_size):
    forces = np.zeros_like(positions)
    for i in range(forces.shape[0]-2):
        for j in range(i+2,forces.shape[0]):
            displacement = positions[j]-positions[i]
            displacement = minimum_image(displacement, box_size)
            distance = np.linalg.norm(displacement)
            if (j-i == 2): #repulsive
                if (distance < 2**(1/6) * sigma): #cutoff
                    epsilon = epsilon_r
                else:
                    continue
            else:
                epsilon = epsilon_a
            magnitude = 24 * epsilon * ((sigma/distance)**12 - 0.5*(sigma/distance)**6) / distance
            force = magnitude * (displacement/distance)
            forces[i] -= force
            forces[j] += force
    return forces

def velocity_verlet(positions, velocities, forces, dt, mass, box_size, ks, r0, epsilon_r, epsilon_a, sigma):
    velocities += 0.5 * forces/mass * dt
    positions += velocities * dt
    positions = apply_pbc(positions, box_size)
    forces_new = compute_harmonic_forces(positions, ks, r0, box_size) + compute_lennard_jones_forces(positions, epsilon_r, epsilon_a, sigma, box_size)
    velocities += 0.5 * forces_new/mass * dt
    return positions, velocities, forces_new

def rescale_velocities(velocities, T0, mass):
    kinetic = 0.5 * mass * np.sum(np.linalg.norm(velocities, axis=1)**2)
    T = 2 * kinetic / (3 * velocities.shape[0])
    scaling_factor = np.sqrt(T0/T)
    velocities *= scaling_factor
    return velocities


def calculate_radius_of_gyration(positions):
    center_of_mass = np.mean(positions, axis=0)
    Rg_squared = np.mean(np.sum((positions - center_of_mass)**2, axis=1))
    Rg = np.sqrt(Rg_squared)
    return Rg

def calculate_end_to_end_distance(positions):
    Ree = np.linalg.norm(positions[-1] - positions[0])
    return Ree

def calculate_potential_energy(positions, box_size, k, r0, epsilon_r, epsilon_a, sigma):
    potential_harmonic = 0.0
    for i in range(positions.shape[0]-1):
        displacement = positions[i]-positions[i+1]
        distance = np.linalg.norm(minimum_image(displacement, box_size))
        potential_harmonic += 0.5*k * (distance - r0)**2
    potential_lj = 0.0
    for i in range(positions.shape[0]-2):
        for j in range(i+2,positions.shape[0]):
            displacement = positions[j]-positions[i]
            distance = np.linalg.norm(minimum_image(displacement, box_size))
            if (j-i == 2): #repulsive
                if (distance < 2**(1/6) * sigma): #cutoff
                    potential_lj += 4 * epsilon_r * ((sigma/distance)**12 - (sigma/distance)**6 + 1/4)
                else:
                    continue
            else: #attractive
                potential_lj += 4 * epsilon_a * ((sigma/distance)**12 - (sigma/distance)**6)
    total = potential_harmonic + potential_lj
    return total


np.random.seed(60)

dt = 0.01  # Time step
total_steps = 10000  # Number of steps
box_size = 100.0  # Size of the cubic box
ks = [1.0, 10.0]  # Spring constant
mass = 1.0  # Particle mass
r0 = 1.0  # Equilibrium bond length
rescale_interval = 100  # Steps between velocity rescaling
n_particles = 20  # Number of particles
epsilon_repulsives = [1.0, 10.0]  # Depth of repulsive LJ potential
epsilon_attractive = 0.5  # Depth of attractive LJ potential
sigma = 1.0  # LJ potential parameter

analysis_values = { # Sorted (k, epsilon_repulsive) for storing Rg, Ree, and E values in that order
    (1.0, 1.0):[],
    (1.0, 10.0):[],
    (10.0, 1.0):[],
    (10.0, 10.0):[]
}

for k in ks:
    for epsilon_repulsive in epsilon_repulsives:
        # Arrays to store properties
        temperatures = np.linspace(0.1, 1.0, 10)
        Rg_values = []
        Ree_values = []
        potential_energies = []


        # Initialize positions and velocities

        for T in temperatures:
            # Set target temperature
            positions = initialize_chain(n_particles, box_size, r0)
            velocities = initialize_velocities(n_particles, T, mass)
            # Simulation loop
            for step in range(total_steps):
                # Compute forces
                forces_harmonic = compute_harmonic_forces(positions, k, r0, box_size)
                forces_lj = compute_lennard_jones_forces(positions, epsilon_repulsive, epsilon_attractive, sigma, box_size)
                total_forces = forces_harmonic + forces_lj
                
                # Integrate equations of motion
                positions, velocities, total_forces = velocity_verlet(positions, velocities, total_forces, dt, mass, box_size, k, r0, epsilon_repulsive, epsilon_attractive, sigma)
                
                # Apply thermostat
                if step % rescale_interval == 0:
                    velocities = rescale_velocities(velocities, T, mass)
            
            nonperiodic_positions = np.zeros_like(positions)
            nonperiodic_positions[0] = positions[0]
            for i in range(1, positions.shape[0]):
                dis = minimum_image(positions[i]-positions[i-1], box_size)
                nonperiodic_positions[i] = (nonperiodic_positions[i-1] + dis)



            Rg = calculate_radius_of_gyration(nonperiodic_positions)
            Ree = calculate_end_to_end_distance(nonperiodic_positions)
            V = calculate_potential_energy(nonperiodic_positions, box_size, k, r0, epsilon_repulsive, epsilon_attractive, sigma)
            Rg_values.append(Rg)
            Ree_values.append(Ree)
            potential_energies.append(V)
            view(ase.Atoms("HeH18He", positions=nonperiodic_positions, velocities=velocities))
        analysis_values[(k, epsilon_repulsive)].append(Rg_values)
        analysis_values[(k, epsilon_repulsive)].append(Ree_values)
        analysis_values[(k, epsilon_repulsive)].append(potential_energies)



# Plotting
plt.figure()
plt.plot(temperatures, analysis_values[(1.0, 1.0)][0], label='k = 1, ε$_r$ = 1')
plt.plot(temperatures, analysis_values[(1.0, 10.0)][0], label='k = 1, ε$_r$ = 10')
plt.plot(temperatures, analysis_values[(10.0, 1.0)][0], label='k = 10, ε$_r$ = 1')
plt.plot(temperatures, analysis_values[(10.0, 10.0)][0], label='k = 10, ε$_r$ = 10')
plt.xlabel('Temperature / k$_B$')
plt.ylabel('Radius of Gyration')
plt.title('Radius of Gyration vs Temperature')
plt.legend()
plt.show()

plt.figure()
plt.plot(temperatures, analysis_values[(1.0, 1.0)][1], label='k = 1, ε$_r$ = 1')
plt.plot(temperatures, analysis_values[(1.0, 10.0)][1], label='k = 1, ε$_r$ = 10')
plt.plot(temperatures, analysis_values[(10.0, 1.0)][1], label='k = 10, ε$_r$ = 1')
plt.plot(temperatures, analysis_values[(10.0, 10.0)][1], label='k = 10, ε$_r$ = 10')
plt.xlabel('Temperature / k$_B$')
plt.ylabel('End-to-End Distance')
plt.title('End-to-End Distance vs Temperature')
plt.legend()
plt.show()

plt.figure()
plt.plot(temperatures, analysis_values[(1.0, 1.0)][2], label='k = 1, ε$_r$ = 1')
plt.plot(temperatures, analysis_values[(1.0, 10.0)][2], label='k = 1, ε$_r$ = 10')
plt.plot(temperatures, analysis_values[(10.0, 1.0)][2], label='k = 10, ε$_r$ = 1')
plt.plot(temperatures, analysis_values[(10.0, 10.0)][2], label='k = 10, ε$_r$ = 10')
plt.xlabel('Temperature / k$_B$')
plt.ylabel('Potential Energy')
plt.title('Potential Energy vs Temperature')
plt.legend()
plt.show()