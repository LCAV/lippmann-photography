from tools import *
from multilayer_optics_matrix_theory import *
import numpy as np
import matplotlib.pyplot as plt


def wavelengths_omega_spaced(lambda_low=400e-9, lambda_high=700e-9, n=300, c0=299792458):
    omega_low = 2 * np.pi * c0 / lambda_high
    omega_high = 2 * np.pi * c0 / lambda_low
    omegas = np.linspace(omega_high, omega_low, n)
    return 2 * np.pi * c0 / omegas


def sigmoid(x, rate=6.0, center=0.5):
    return 1 / (1 + np.exp(-rate * (x - center)))


def simulate_printing(blocks, block_height, block_length, max_index_change, z, base_index, lambdas):
    """scale might be a vector of different heights, delta_n is a scalar"""

    # transform to low level description
    distances, index_in_blocks = blobs_to_matrices(
        begs=blocks,
        delta_n=block_height,
        ends=(blocks + block_length),
        n0=0)
    index_in_blocks = sigmoid(index_in_blocks) * max_index_change + base_index

    index_unif_spaced = blobs_to_ref_index(blob_z0=blocks,
                                           blob_delta_z=block_length,
                                           n0=0,
                                           delta_n=block_height,
                                           depths=z)
    index_unif_spaced = sigmoid(index_unif_spaced) * max_index_change + base_index

    # calculate reflection:
    reflection, _ = propagation_arbitrary_layers_Born_spectrum(index_in_blocks,
                                                               d=distances,
                                                               lambdas=lambdas,
                                                               plot=False)

    return reflection, index_unif_spaced


if __name__ == '__main__':
    # setting process related constants
    n0 = 1.45
    delta_n = 1e-3
    dot_size = 0.2e-6
    expected_wavelength = 510e-9
    depth = 100e-6

    # frequency discretization
    wavelengths = wavelengths_omega_spaced()

    # create grating:
    step_size = expected_wavelength / (2.0 * n0)
    dots = np.arange(0, depth, step_size)

    # choose model (small scale - many possible heights, scale close to 1 - one possible height)
    scale = 0.5

    # choose the resolution at which the pattern will be plotted
    depths = np.linspace(0, depth, 10000)

    # actually calculate things
    R, draw_index = simulate_printing(blocks=dots,
                                      block_height=scale,
                                      block_length=dot_size,
                                      max_index_change=delta_n,
                                      z=depths,
                                      base_index=n0,
                                      lambdas=wavelengths)

    # plot the index of refraction (only first part)
    range_to_plot = 100
    plt.figure()
    plt.plot(depths[:100], draw_index[:100])
    plt.xlabel("refractive index", fontsize=20)

    # plot the reflection
    plt.figure()
    plt.plot(wavelengths, R, linewidth=2)
    plt.xlabel(r"$\lambda$", fontsize=20)
    plt.ylabel("reflection", fontsize=20)
    plt.show()
