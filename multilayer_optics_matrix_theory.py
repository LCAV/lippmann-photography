# -*- coding: utf-8 -*-
"""
Created on Thu May 18 19:12:57 2017

@author: gbaechle
"""

import numpy as np
import scipy as sp
import scipy.stats
import matplotlib.pyplot as plt
import math

plt.close('all')


# can be used interchangably from S to M or from M to S
def from_S_to_M(S):
    return 1 / S[1, 1] * np.array([[S[0, 0] * S[1, 1] - S[0, 1] * S[1, 0], S[0, 1]], [-S[1, 0], 1]])


def Fresnel_equations(n1, n2, theta1, polarized='s'):
    cos_theta1 = np.cos(theta1)
    cos_theta2 = np.sqrt(1 - (n1 / n2) ** 2 * np.sin(theta1) ** 2)

    if polarized == 's' or polarized == 'TE':
        r = (n1 * cos_theta1 - n2 * cos_theta2) / (n1 * cos_theta1 + n2 * cos_theta2)
        t = 1 + r
        return r, t

    else:
        sec_theta1 = 1 / cos_theta1
        sec_theta2 = 1 / cos_theta2
        r = (n1 * sec_theta1 - n2 * sec_theta2) / (n1 * sec_theta1 + n2 * sec_theta2)
        t = (1 + r) * cos_theta1 / cos_theta2
        return r, t


def propagation_followed_by_boundary(n1, n2, phi):
    one_over_t = (n2 + n1) * np.exp(1j * phi)
    r_over_t = (n2 - n1) * np.exp(1j * phi)

    M = 1 / (2 * n2) * np.array([[np.conj(one_over_t), r_over_t], [np.conj(r_over_t), one_over_t]])
    return M


def propagation_arbitrary_layers(ns, k, d):
    # start with an interface with air
    #    n1 = 1
    #    n2 = ns[0]
    #    M = 1/(2*n2)*np.array([[n2+n1, n2-n1], [n2-n1, n2+n1]])

    M = np.eye(2)

    for i in range(len(ns) - 1):
        n1 = ns[i]
        n2 = ns[i + 1]
        phi = n1 * k * d
        Mi = propagation_followed_by_boundary(n1, n2, phi)

        M = M @ Mi

    S = from_S_to_M(M)

    t = S[0, 0]
    r = S[0, 1]

    return r, t


def dielectric_Brag_grating(N, n1, n2, phi1, phi2):
    M1 = propagation_followed_by_boundary(n1, n2, phi1)
    M2 = propagation_followed_by_boundary(n2, n1, phi2)

    M0 = M2 @ M1

    M = np.linalg.matrix_power(M0, N)

    S = from_S_to_M(M)

    t = S[0, 0]
    r = S[0, 1]

    return r, t


def propagation_arbitrary_layers_spectrum(ns, d, lambdas, plot=True, symmetric=False):
    ks = 2 * np.pi / lambdas
    if not hasattr(d, "__iter__"):
        d = np.ones_like(lambdas) * d

    if symmetric:
        n = np.r_[ns[:0:-1], ns]
        dist = np.r_[d[:0:-1], d]
    else:
        n = ns
        dist = d

    total_reflectance = []
    total_transmittance = []

    for (k, distance) in zip(ks, dist):
        r, t = propagation_arbitrary_layers(n, k, distance)
        total_reflectance.append(np.abs(r) ** 2)
        total_transmittance.append(np.abs(t) ** 2)

    if plot:
        plt.figure()
        plt.plot(lambdas, total_reflectance)
        plt.title('Reflected spectrum with transmission matrices')
        plt.draw()

    return total_reflectance, total_transmittance


def dielectric_Brag_grating_spectrum(N, n1, n2, d1, d2, lambdas, plot=True):
    k = 2 * np.pi / lambdas

    phis1 = n1 * k * d1
    phis2 = n2 * k * d2

    total_reflectance = []
    total_transmittance = []

    for (phi1, phi2) in zip(phis1, phis2):
        r, t = dielectric_Brag_grating(N, n1, n2, phi1, phi2)
        total_reflectance.append(np.abs(r) ** 2)
        total_transmittance.append(np.abs(t) ** 2)

    if plot:
        plt.figure()
        plt.plot(lambdas, total_reflectance)
        plt.draw()

    return lambdas, total_reflectance, total_transmittance


def propagation_Born(n, phi):
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    M = np.array([[cos_phi, -1j / n * sin_phi], [-1j * n * sin_phi, cos_phi]])
    return M
    
    
def propagation_Lippmann_matrix(r, d, n, k0, epsilon=0.2E7):
    
    phi = n*k0*d

    r *= d*epsilon
#    r = np.sqrt(r) 
    
    t = np.sqrt(1-r**2)
#    M = np.array([[1, 1j*r], [-1j*r, 1]])/t
    
    S = np.array([[t,1j*r], [1j*r,t]])
    M = from_S_to_M(S)
    P = np.array([[np.exp(-1j * phi), 0], [0, np.exp(1j * phi)]])
    
    return M @ P
    
    
def propagation_arbitrary_layers_Lippman(rs, ds, k, n):

    M = np.eye(2)

    for r, d in zip(rs, ds):
        ()
        Mi = propagation_Lippmann_matrix(r, d, n, k)
        M = Mi @ M
        

    S = from_S_to_M(M)

    t = S[0, 0]
    r = S[0, 1]

    return r, t


def propagation_arbitrary_layers_Lippmann_spectrum(rs, d, lambdas, n0=1.45, plot=True, symmetric=False):
    ks = 2 * np.pi / lambdas
    if not hasattr(d, "__iter__"):
        d = np.ones_like(rs) * d

    if symmetric:
        refl = np.r_[rs[:0:-1], rs]
        dist = np.r_[d[:0:-1], d]
    else:
        refl = rs
        dist = d

    total_reflectance = []
    total_transmittance = []

    for k in ks:
        r, t = propagation_arbitrary_layers_Lippman(refl, dist, k, n0)
        total_reflectance.append(np.abs(r) ** 2)
        total_transmittance.append(np.abs(t) ** 2)

    if plot:
        plt.figure()
        plt.plot(lambdas, total_reflectance)
        plt.title('Reflected spectrum with Lippmann matrices')
        plt.draw()

    return total_reflectance, total_transmittance


def propagation_arbitrary_layers_Born(ns, k, d):
    M = np.eye(2)

    if not hasattr(d, "__iter__"):
        d = np.ones_like(ns) * d

    for (n, dist) in zip(ns, d):
        phi = n * k * dist
        Mi = propagation_Born(n, phi)

        M = M @ Mi

    p_1 = ns[0]
    p_ell = ns[-1]
    m1 = (M[0, 0] + M[0, 1] * p_ell) * p_1 - (M[1, 0] + M[1, 1] * p_ell)
    m2 = (M[0, 0] + M[0, 1] * p_ell) * p_1 + (M[1, 0] + M[1, 1] * p_ell)

    r = m1 / m2
    t = 2 * p_1 / m2

    return np.abs(r) ** 2, ns[-1] / ns[0] * np.abs(t) ** 2


def propagation_arbitrary_layers_Born_spectrum(ns, d, lambdas, plot=True):
    ks = 2 * np.pi / lambdas

    total_reflectance = []
    total_transmittance = []

    for k in ks:
        R, T = propagation_arbitrary_layers_Born(ns, k, d)
        total_reflectance.append(R)
        total_transmittance.append(T)

    if plot:
        plt.figure()
        plt.plot(lambdas, total_reflectance)
        plt.title('Reflected spectrum with transmission matrices (Born method)')
        plt.draw()

    return total_reflectance, total_transmittance


def generate_gaussian_spectrum(lambdas, mu=550E-9, sigma=30E-9):
    spectrum = sp.stats.norm(loc=mu, scale=sigma).pdf(lambdas)

    return spectrum


def generate_rect_spectrum(lambdas, start=450E-9, end=560E-9):
    spectrum = np.zeros(len(lambdas))
    spectrum[(lambdas >= start) & (lambdas <= end)] = 1

    return spectrum


def generate_mono_spectrum(lambdas, color=550E-9):
    spectrum = np.zeros(len(lambdas))
    spectrum[np.argmin(np.abs(lambdas - color))] = 1

    return spectrum


def lippmann_transform(lambdas, spectrum, depths):
    """"Compute the Lippmann transform

        lambdas     - vector of wavelengths
        spectrum    - spectrum of light
        depths      - vector of depths

        Returns intensity       - computed intensity of the interfering waves
                delta_intensity - the intensity without the baseline term"""""
        
    two_k = 4 * np.pi / lambdas

    one_minus_cosines = 0.5 * (1 - np.cos(two_k[None, :] * depths[:, None]))
    cosines = 0.5 * np.cos(two_k[None, :] * depths[:, None])

    intensity = -np.trapz(one_minus_cosines * spectrum[None, :], two_k, axis=1)
    delta_intensity = np.trapz(cosines * spectrum[None, :], two_k, axis=1)
    return intensity, delta_intensity


def inverse_lippmann(intensity, lambdas, depths, symmetric=False):
    two_k = 4 * np.pi / lambdas
    
    if symmetric:
        I = np.r_[intensity[:0:-1], intensity]
        d = np.r_[-depths[:0:-1], depths]
    else:
        I = intensity
        d = depths
    
    exponentials = np.exp(-1j * two_k[:, None] * d[None, :])
    return np.abs(1/d[-1] * np.trapz(exponentials * I[None, :], d, axis=1)) ** 2


def generate_lippmann_refraction_indices(delta_intensity, n0=1.45, k0=0., mu_n=0.1, mu_k=0., s0=1):
    """"Generates indices of refraction given a vector of intensity variations

        delta_intensity - vector of intensity variations
        n0              - basic index of refraction
        k0              - basic attenuation coefficient (imaginary part of index of refraction)
        mu_n            - maximum change in index of refraction
        mu_k            - maximum change in attenuation coefficient
        s0              - clipping level

        Returns n - list of indices corresponding to depths depths"""""    
    
    s = delta_intensity / np.max(delta_intensity)
    s[s > s0] = s0

    n = n0 * (1 + mu_n * s) + 1j * k0 * (1 + mu_k * s)
    return n


def blobs_to_ref_index(blob_z0, blob_delta_z, n0, delta_n, depths):
    """"Calculates the index of refraction at depths `depths`

        blob_z0         - array of beginnings of the blobs
        blob_delta_x    - size of the blob
        n0              - basic index of refraction
        delta_n         - change of index of refraction per blob (scalar)
        depths          - uniform array of distances

        Returns n - list of indices corresponding to depths depths"""""

    n = np.ones_like(depths) * n0
    delta_z = depths[1] - depths[0]
    for z0 in blob_z0:
        idx_min = math.floor(z0 / delta_z)
        idx_max = math.floor((z0 + blob_delta_z) / delta_z)
        n[idx_min:idx_max] = n[idx_min:idx_max] + delta_n
    return n


def blobs_to_matrices(begs, ends, n0, delta_n):
    """"Calculates set of distances and refractive indices
        begs    - beginnings of blobs
        ends    - ends of blobs
        n0      - basic index of refraction
        delta_n - change of index of refraction per blob (scalar or array)

        Returns detla_z - list of distances and n - list of corresponding indices"""""

    if not hasattr(delta_n, "__iter__"):
        delta_n = np.ones_like(begs) * delta_n

    delta_z = []
    n = []
    beg = 0
    curr_d_n = n0
    curr_d_z = 0
    for end in range(len(ends)):
        while beg < len(begs) and begs[beg] < ends[end]:
            # create matrix
            delta_z.append(begs[beg] - curr_d_z)
            n.append(curr_d_n)
            # update stuff
            curr_d_z = begs[beg]
            curr_d_n += delta_n[beg]
            beg += 1
            # create matrix
        delta_z.append(ends[end] - curr_d_z)
        n.append(curr_d_n)
        # update stuff
        curr_d_z = ends[end]
        curr_d_n -= delta_n[end]

    return np.array(delta_z), np.array(n)


if __name__ == '__main__':
#    plt.ion()

    N = 1000
    n1 = 1.45
    #    n2 = 1.451
    n2 = 1.8
    d1 = 20E-9
    d2 = 20E-9

    ns = [n1, n2] * N
    ns = np.random.rand(N) * 0.1 + 1.45
    ns = np.random.randn(N) * 0.1 + 1.45
    ns = np.linspace(1.45, 1.6, N)

    #    dielectric_Brag_grating_spectrum(N, n1, n2, d1, d2)
    #    propagation_arbitrary_layers_spectrum(ns, d=d1)

    n0 = 1.45
    c0 = 299792458
    c = c0 / n0
    N_omegas = 300
    delta_z = 10E-9
    max_depth = 5E-6

    lambda_low = 390E-9;
    lambda_high = 700E-9
    omega_low = 2 * np.pi * c0 / lambda_high;
    omega_high = 2 * np.pi * c0 / lambda_low
    omegas = np.linspace(omega_high, omega_low, 300)
    lambdas = 2 * np.pi * c0 / omegas

    spectrum = generate_gaussian_spectrum(lambdas=lambdas, mu=550E-9, sigma=30E-9)
#    spectrum = generate_rect_spectrum(lambdas=lambdas, start=450E-9, end=540E-9)
#    spectrum = generate_mono_spectrum(lambdas, color=440E-9)

    plt.figure()
    plt.plot(lambdas, spectrum)
    plt.title('Original object spectrum')

    depths = np.arange(0, max_depth, delta_z)
    intensity, delta_intensity = lippmann_transform(lambdas / n0, spectrum, depths)
    
#    intensity = np.ones_like(depths)
#    delta_intensity = np.ones_like(depths)
    
    ns = generate_lippmann_refraction_indices(delta_intensity, n0=n0, mu_n=0.01)

    plt.figure()
    plt.plot(depths, intensity)
    plt.title('Lippmann transform')

    #    plt.figure()
    #    plt.plot(depths, ns)
    #    plt.show()
    #    plt.title('Lippmann refraction coefficients')

    inverted_lippmann = inverse_lippmann(intensity, lambdas / n0, depths)
    plt.figure()
    plt.plot(lambdas, inverted_lippmann)
    plt.title('Inverse Lippmann transform')

    inverted_lippmann = inverse_lippmann(intensity, lambdas / n0, depths, symmetric=True)
    plt.figure()
    plt.plot(lambdas, inverted_lippmann)
    plt.title('Inverse Lippmann transform (symmetric)')
    
    r, t = propagation_arbitrary_layers_Lippmann_spectrum(rs=intensity/np.max(intensity), d=delta_z, lambdas=lambdas)

    propagation_arbitrary_layers_spectrum(ns, d=delta_z, lambdas=lambdas, symmetric=True)
    propagation_arbitrary_layers_Born_spectrum(ns, d=delta_z, lambdas=lambdas)
#    plt.ioff()
    plt.show()
