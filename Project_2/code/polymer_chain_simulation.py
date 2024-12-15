import numpy as np
from scipy.stats import maxwell
import matplotlib.pyplot as plt

# Temperature in terms of k_B

def boltzmann_distribution(v, mass, T):
    return np.pi * (mass / (2 * np.pi * T))**(3/2) * v**2 * np.exp(-mass * v**2 / (2 * T))

def T_instant(N_particles, velocities):
    KE = 2

def minimum_image(delta, box_size):
    displacement = delta - (box_size * np.round((delta) / box_size))
    return displacement


def initialize_chain(N_particles, box_size, r0):
    positions = np.zeros((N_particles,3))
    current = [box_size/2,box_size/2,box_size/2]
    positions[0] = current
    for i in range(1,N_particles):
        theta = np.pi*np.random.random()
        phi = 2*np.pi*np.random.random()
        direction = [np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)]
        next = current + r0*direction
        positions[i] = apply_pbc(next, box_size)
    return positions

def initialize_velocities(N_particles, T0, mass):
    velocities = np.zeros((N_particles,3))
    for i in range(N_particles):
        theta = np.pi*np.random.random()
        phi = 2*np.pi*np.random.random()
        direction = [np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)]
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
        force = magnitude * displacement / distance
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
                    magnitude = 4*epsilon_r * ((sigma/distance)**12 - (sigma/distance)**6 + 1/4)
                    force = magnitude * (displacement/distance)
                    forces[i] -= force
                    forces[j] += force
                continue
            else:
                magnitude = 4*epsilon_a * ((sigma/distance)**12 - (sigma/distance)**6)
                force = magnitude * (displacement/distance)
                forces[i] -= force
                forces[j] += force
    return forces














if (__name__ == "__main__"):
    sigma = 1.0
    N = 20
    epsilon_a = 0.5
    epsilon_r = 1.0
    r0 = 1.0
    T0 = 0.1