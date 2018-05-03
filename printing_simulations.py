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


def sigmoid_inverse(y, min_margin=0.05, max_margin=0.95):
    y = y - np.min(y)
    y = y * (max_margin - min_margin) / (np.max(y))
    y = y + min_margin
    return np.log((y / (1 - y)))


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


def block_approximate(depths, values, block_size, scale, mass=True):
    blocks = []
    intensities = []
    for idx in range(len(values)-block_size):
        if mass:
            multiple = math.floor(np.sum(values[idx:idx + block_size]) / (scale * block_size))
        else:
            multiple = math.floor(np.min(values[idx:idx + block_size]) / scale)
        if multiple > 0:
            tmp_f = values[idx:idx + block_size] - multiple * scale
            tmp_f[tmp_f < 0.0] = 0.0
            values[idx:idx + block_size] = tmp_f
            blocks.append(depths[idx])
            intensities.append(multiple * scale)
    if len(blocks) < 1:
        raise Warning('No blocks were created during approximation!')
    return np.array(blocks), np.array(intensities)


if __name__ == '__main__':
    # setting process related constants
    n0 = 1.45
    delta_n = 1e-3
    dot_size = 0.2e-6
    expected_wavelength = 510e-9
    depth = 100e-6
    save = True

    # frequency discretization
    wavelengths = wavelengths_omega_spaced(n=500)

    # space discretization
    depths = np.linspace(0, depth, 0.5e5)
    depth_res = depths[1]-depths[0]

    # choose model (small scale - many possible heights, scale close to 1 - one possible height)
    block_scale = 0.1
    pattern_scale = 2

    # generate pattern:
    # spectrum = generate_mono_spectrum(wavelengths, color=expected_wavelength)
    spectrum = generate_gaussian_spectrum(lambdas=wavelengths, mu=expected_wavelength, sigma=10e-9)
    _, delta_intensity = lippmann_transform(wavelengths / n0, spectrum, depths)
    delta_intensity = sigmoid_inverse(delta_intensity)
    blocks, intensities = block_approximate(depths, pattern_scale * delta_intensity,
                                            math.floor(dot_size / depth_res), block_scale)


    # actually calculate things
    R, draw_index = simulate_printing(blocks=blocks,
                                        block_height=intensities,
                                        block_length=dot_size,
                                        max_index_change=delta_n,
                                        z=depths,
                                        base_index=n0,
                                        lambdas=wavelengths)

    # plot the index of refraction (only first part)
    range_to_plot = 1000
    plt.figure()
    plt.plot(depths[:range_to_plot], draw_index[:range_to_plot])
    plt.xlabel("depth", fontsize=20)
    plt.ylabel("refractive index", fontsize=20)
    plt.ylim([n0, n0 + delta_n*1.1])

    # plot the reflection
    plt.figure()
    plt.plot(wavelengths, R, linewidth=2)
    plt.xlabel(r"$\lambda$", fontsize=20)
    plt.ylabel("reflection", fontsize=20)
    plt.show()

    # how to save file?
    code = "simulation_plots/gaussian_mass_scale_"+str(block_scale)
    if save:
        print("saving files in: ", code)
        np.save(code+"_depths", depths)
        np.save(code+"_idex1", draw_index)
        np.save(code+"_wavels", wavelengths)
        np.save(code+"_reflection1", R)
