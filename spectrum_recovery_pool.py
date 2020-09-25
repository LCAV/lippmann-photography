#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 08:03:24 2018

@author: gbaechle
"""
import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

import glob
from multiprocessing import Pool
import pickle
from scipy.optimize import fmin, nnls
import time


from lippmann2 import *
import display_spectral_data as dsd
from safe_save import *

c0 = 299792458
n0 = 1.5
c = c0 / n0


def generate_matrix_F(omegas, Z, margin=0.5):
    omegas_sinc = generate_wavelengths_sinc(Z, omegas, c, margin=margin)[1]

    mi = np.min(omegas)
    ma = np.max(omegas)

    T = np.abs(omegas_sinc[1] - omegas_sinc[0])
    sinc = np.tile(omegas, (len(omegas_sinc), 1)) - np.tile(omegas_sinc[:, None], (1, len(omegas)))

    s = np.sinc(sinc / T).T

    if margin <= 0:
        return s

    e_idx = np.where(omegas_sinc < mi)[0]
    b_idx = np.where(omegas_sinc > ma)[0]

    s_cropped = s[:, np.max(b_idx) + 1: np.min(e_idx)]
    s_cropped[:, 0] += np.sum(s[:, b_idx], axis=1)
    s_cropped[:, -1] += np.sum(s[:, e_idx], axis=1)

    return s_cropped


def generate_matrix_A(omegas, Z, r=-1, mode=1, k0=0, over_samples=0):
    lambdas = 2 * np.pi * c / omegas

    mi = np.min(omegas)
    ma = np.max(omegas)
    delta = np.abs(omegas[0] - omegas[1])

    if over_samples == 0:
        omegas_over = omegas
    else:
        omegas_over = np.arange(ma + over_samples * delta, mi - over_samples * delta, -delta)
    lambdas_over = 2 * np.pi * c / omegas_over

    A = np.zeros((len(omegas), len(omegas_over)), dtype=complex)
    A0 = np.zeros((len(omegas), len(omegas)), dtype=complex)

    for i, lambda_prime in enumerate(lambdas):
        A[i, :] = fda.h(lambdas_over, lambda_prime, Z, r=r, mode=2, k0=k0)
        A0[i, :] = fda.h(lambdas_over, lambda_prime, Z, r=r, mode=3, k0=k0)

    if over_samples == 0:
        A_cropped = A
    else:
        e_idx = np.where(omegas_over <= mi)[0][1:]
        b_idx = np.where(omegas_over > ma)[0]

        A_cropped = A[:, np.max(b_idx) + 1: np.min(e_idx)]
        A_cropped[:, 0] += np.sum(A[:, b_idx], axis=1)
        A_cropped[:, -1] += np.sum(A[:, e_idx], axis=1)

    if mode == 1:
        A_cropped += A0
    elif mode == 3:
        A_cropped = A0

    # normalization
    return 4 / np.pi * np.abs(omegas[1] - omegas[0]) / c * A_cropped


def generate_matrix_B(omegas, Z, r=-1, mode=1, k0=0, margin=0.5):
    lambdas = 2 * np.pi * c / omegas
    lambdas_sinc, omegas_sinc = generate_wavelengths_sinc(Z, omegas, c, margin=margin)

    mi = np.min(omegas)
    ma = np.max(omegas)

    B = np.zeros((len(lambdas), len(lambdas_sinc)), dtype=complex)
    B_edges = np.zeros((len(lambdas), len(lambdas_sinc)), dtype=complex)
    for i, lambda_prime in enumerate(lambdas):
        B[i, :] = fda.h(lambdas_sinc, lambda_prime, Z, r=r, mode=1, k0=k0)
        B_edges[i, :] = fda.h(lambdas_sinc, lambda_prime, Z, r=r, mode=2, k0=k0)

    B *= 4 / np.pi * np.abs(omegas_sinc[1] - omegas_sinc[0]) / c
    B_edges *= 4 / np.pi * np.abs(omegas_sinc[1] - omegas_sinc[0]) / c

    if margin <= 0:
        return B

    e_idx = np.where(omegas_sinc < mi)[0]
    b_idx = np.where(omegas_sinc > ma)[0]

    B_cropped = B[:, np.max(b_idx) + 1: np.min(e_idx)]
    B_cropped[:, 0] += np.sum(B_edges[:, b_idx], axis=1)
    B_cropped[:, -1] += np.sum(B_edges[:, e_idx], axis=1)

    return B_cropped


def generate_matrix_B_(omegas, Z, r=-1, mode=1, k0=0):
    lambdas = 2 * np.pi * c / omegas
    lambdas_sinc, omegas_sinc = generate_wavelengths_sinc(Z, omegas, c)
    #    lambdas_sinc, omegas_sinc = lambdas, omegas

    B = np.zeros((len(lambdas), len(lambdas_sinc)), dtype=complex)
    for i, lambda_prime in enumerate(lambdas):
        for j, lambda_sinc in enumerate(lambdas_sinc):
            om_max = 2 * np.pi * c / 250E-9
            om = np.linspace(om_max, 0, 1000)
            lam = 2 * np.pi * c / om

            omega_sinc = 2 * np.pi * c / lambda_sinc
            T = np.abs(omegas_sinc[1] - omegas_sinc[0])
            sinc = np.sinc((om - omega_sinc) / T)
            B[i, j] = -np.trapz(sinc * fda.h(lam, lambda_prime, Z, r=r, mode=mode, k0=k0), om)

    return B


def generate_wavelengths_sinc(Z, omegas, c=c, margin=0):
    if Z == 'inf' or Z == 'infinite':
        Z = 100E-6  # default behavior to avoid problems

    mi = np.min(omegas) * (1 - margin)
    ma = np.max(omegas) * (1 + margin)

    Z *= 2

    if mi < 0:
        omegas_sinc_p = np.arange(0, ma, np.pi * c / Z)
        omegas_sinc_n = np.arange(0, mi, -np.pi * c / Z)[::-1]
        omegas_sinc = np.r_[omegas_sinc_n[:-1], omegas_sinc_p]
    else:
        omegas_sinc = np.arange(np.pi * c / (2 * Z), ma, np.pi * c / Z)
        omegas_sinc = omegas_sinc[omegas_sinc >= mi]

    omegas_sinc = omegas_sinc[::-1]
    lambdas_sinc = 2 * np.pi * c / omegas_sinc

    return lambdas_sinc, omegas_sinc


def power_spectrum_from_complex_wave(omegas, replay, r, Z, k0):
    F = generate_matrix_F(omegas, Z)
    A = generate_matrix_A(omegas, Z, r=r, k0=k0)

    Bc = A @ F
    B = np.r_[np.real(Bc), np.imag(Bc)]

    try:
        spectrum = nnls(B, np.r_[np.real(replay), np.imag(replay)])[0]
    except RuntimeError:
        print("Oops! RuntimeError using nonnegative least squares, using standard one instead")
        spectrum = np.linalg.lstsq(B, np.r_[np.real(replay), np.imag(replay)], rcond=None)[0]
        spectrum = np.maximum(spectrum, 0)

    return spectrum


def project_onto_subspace(omegas, spectrum, r, Z, k0):
    F = generate_matrix_F(omegas, Z)
    A = generate_matrix_A(omegas, Z, r=r, k0=k0)
    Bc = A @ F
    #    Bc = generate_matrix_B(omegas, Z, r=r, k0=k0)
    B = np.r_[np.real(Bc), np.imag(Bc)]

    return Bc @ nnls(B, np.r_[np.real(replay), np.imag(replay)])[0]


def forward_model(omegas, spectrum, r, Z, k0):
    F = generate_matrix_F(omegas, Z)
    A = generate_matrix_A(omegas, Z, r=r, k0=k0)
    B = A @ F
    return B @ spectrum


def spectrum_recovery(omegas, signal_measured, r=-1, Z=6E-6, k0=0, infinite=True, n_iter=200, plot=False,
                      estimate_depth=True):
    """Used in spectrum_recovery_data"""

    lambdas = 2 * np.pi * c / omegas
    signal_est = None

    if estimate_depth:
        k0 = 3.
        Z = estimate_Z(lambdas, signal_measured)
    #        Z = 17.2066E-6
    #        Z = 9.6E-6

    Z_lb = Z
    print(Z, k0)

    F = generate_matrix_F(omegas, Z)
    signal_est = np.ones(F.shape[1])
    complex_wave = forward_model(omegas, signal_est, r, Z, k0)

    print(f"starting {n_iter} iterations")
    for i in range(n_iter):

        complex_wave_prev = np.copy(complex_wave)
        complex_wave = forward_model(omegas, signal_est, r, Z, k0)
        #        complex_wave = project_onto_subspace(omegas, complex_wave, r, Z, k0)

        if plot and np.mod(i, n_iter // 20) == 0:
            signal_est = power_spectrum_from_complex_wave(omegas, complex_wave, r, Z, k0)
            lambdas_sinc, omegas_sinc = generate_wavelengths_sinc(Z, omegas, c)

            plt.figure(figsize=(5, 5))
            plt.plot(lambdas, np.real(complex_wave_prev))
            plt.plot(lambdas, np.real(complex_wave), 'k--')
            plt.plot(lambdas, np.imag(complex_wave_prev))
            plt.plot(lambdas, np.imag(complex_wave), 'k--')

            plt.figure(figsize=(5, 5))
            plt.fill_between(lambdas, -np.sqrt(signal_measured), np.sqrt(signal_measured), alpha=0.2)
            plt.plot(lambdas, np.abs(complex_wave), 'r')
            plt.plot(lambdas, np.real(complex_wave), 'r--')
            plt.plot(lambdas, np.imag(complex_wave), 'r:')
            plt.plot(lambdas, generate_matrix_F(omegas, Z) @ signal_est, 'k')
            plt.plot(lambdas_sinc, signal_est, 'k--')
            plt.title('i = ' + str(i))
            plt.show()

        complex_wave *= np.sqrt(signal_measured) / np.abs(complex_wave)

        if estimate_depth:
            Z2, k0 = refine_Z_tau(i, lambdas, complex_wave, signal_measured, Z, r, k0, lb=Z_lb)

        signal_est = power_spectrum_from_complex_wave(omegas, complex_wave, r, Z, k0)

    #    return fda.sinc_interp(power_spectrum_from_complex_wave(omegas, complex_wave, r, Z, k0), omegas_sinc, omegas)
    return generate_matrix_F(omegas, Z) @ power_spectrum_from_complex_wave(omegas, complex_wave, r, Z, k0), Z, k0


def estimate_Z(lambdas, signal):
    """Used in spectrum_recovery"""
    n_freq = 10000
    omegas = 2 * np.pi * c / lambdas
    delta_omega = omegas[0] - omegas[1]

    fft = np.fft.fft(signal, n_freq)
    Zs = np.pi * np.fft.fftfreq(n_freq, delta_omega) * c

    fft[Zs < 2.5E-6] = 0
    fft[Zs > 50E-6] = 0

    Z_est = Zs[np.argmax(np.abs(fft))]
    return Z_est


def refine_Z_tau(i, lambdas, complex_wave, signal_measured, Z, r, k0, lb=0):
    omegas = 2 * np.pi * c / lambdas

    signal_est = generate_matrix_F(omegas, Z) @ power_spectrum_from_complex_wave(omegas, complex_wave, r, Z, k0)
    f = lambda x: generate_matrix_A(omegas, Z=x[0], r=r, mode=1, k0=x[1]) @ signal_est
    #    f = lambda Z_est, tau_est: project_onto_subspace(omegas, complex_wave, r, Z_est, k0)
    errorZ = lambda Z_est: np.sum((np.abs(f([Z_est, k0])) - np.sqrt(signal_measured)) ** 2)
    errortau = lambda tau_est: np.sum((np.abs(f([Z, tau_est])) - np.sqrt(signal_measured)) ** 2)

    Z = np.maximum(lb, Z)

    # get out of local minima
    if i < 3 or np.mod(i, 50) == 0:

        Zs = np.linspace(np.maximum(1E-6, Z - 3E-6), Z + 3E-6, 201)
        errorZ = lambda Z_est: np.sum((np.abs(f([Z_est, k0])) - np.sqrt(signal_measured)) ** 2)
        errors = [errorZ(zi) for zi in Zs]
        Z = Zs[np.argmin(errors)]

        taus = np.linspace(0, 5, 201)
        errors = [errortau(tau) for tau in taus]
        k0 = taus[np.argmin(errors)]

        print(Z, k0)
    elif np.mod(i, 5) == 0:

        error = lambda x: np.sum((np.abs(f(x)) - np.sqrt(signal_measured)) ** 2)
        xopt = fmin(error, x0=[Z, k0], disp=False)
        Z, k0 = np.abs(xopt[0]), np.maximum(0, xopt[1])
        #        Z = xopt[0]

        print(Z, k0)

    return Z, k0


def spectrum_recovery_spectrometer(path, file, N=200, r='hg', visible=True, normalize=True, n_iter=200):
    if r == 'hg':
        r = 0.7 * np.exp(1j * np.deg2rad(148))

    wavelengths, data, i_time = dsd.read_file(path + file)

    if normalize:
        white_path = path + 'Bright frame ' + str(int(i_time)) + '.txt'
        black_path = path + 'Dark frame ' + str(int(i_time)) + '.txt'
        data_norm = dsd.normalize_data(wavelengths, data, i_time, white_path, black_path)
    else:
        data_norm = data
    data_norm = np.nan_to_num(data_norm)
    data_norm = np.maximum(0, data_norm)

    spectrum_est, Z_est, k0_est, spec_lp = spectrum_recovery_data(wavelengths, data_norm, N=N, r=r, visible=visible,
                                                                  Z=30E-6, n_iter=n_iter)

    make_dirs_safe(np.save, 'Data/Nature/lambdas', wavelengths)
    np.save('Data/Nature/spectrum_est_' + file[:-4], spec_lp)
    np.save('Data/Nature/Z_est_' + file[:-4], Z_est)
    np.save('Data/Nature/k0_est_' + file[:-4], k0_est)

    return spectrum_est, Z_est, k0_est


def spectrum_recovery_data(lambdas_nu, spectrum_nu, N=200, r=0.2, visible=True, Z=8E-6, n_iter=200):
    """ Used in the main script"""
    k0 = 4.5

    omegas_nu = 2 * np.pi * c / lambdas_nu
    if visible:
        omegas = np.linspace(2 * np.pi * c / 400E-9, 2 * np.pi * c / 700E-9, N)
    else:
        omegas = np.linspace(np.max(omegas_nu), np.min(omegas_nu), N)
    spectrum = sp.interpolate.interp1d(omegas_nu, spectrum_nu, kind='cubic', bounds_error=False,
                                       fill_value='extrapolate')(omegas)

    spectrum = np.maximum(0, spectrum)

    spectrum_est, Z_est, k0_est = spectrum_recovery(omegas, spectrum, r=r, Z=Z, k0=k0, n_iter=n_iter, plot=False,
                                                    estimate_depth=False)

    return spectrum_est, Z_est, k0_est, spectrum_est


if __name__ == '__main__':

    plt.close('all')

    params = {"N": 100,
              "Z": 5E-6,
              "k0": 1,
              "r":  0.7 * np.exp(1j * np.deg2rad(148)),
              "n_iter": 200,
              "visible": True}

    directory = 'Cubes/'

    for file in glob.glob(directory + '/*.dat'):
        name = file.split('/')[-1][:-4]
        print("\n")
        print("processing file", name)

        data, wavelengths = dsd.load_specim_data(directory + name, ds=25, cut=True)

        start = time.time()

        def spectrum_recovery_row(x):
            inverted_row = np.zeros((data.shape[1], params["N"]))
            row_time = time.time()
            for y in range(data.shape[1]):
                pixel_time = time.time()
                print(f"processing ({x}, {y})")
                spectrum_est, Z_est, k0_est, spec_lp = \
                    spectrum_recovery_data(wavelengths,
                                           data[x, y, :],
                                           N=params["N"],
                                           r=params["r"],
                                           visible=params["visible"],
                                           Z=params["Z"],
                                           n_iter=params["n_iter"])
                inverted_row[y, :] = spectrum_est
                print(f"Pixel time: {time.time() - pixel_time}")
            print(f"Row time: {time.time() - row_time}")
            return inverted_row

        pool = Pool(2)
        inverted = np.array(pool.map(spectrum_recovery_row, range(data.shape[0])))

        print(f"Time elapsed: {time.time() - start}")

        i = 0
        dirname = f"PNAS/{i}/"
        if not os.path.exists("PNAS"):
            os.mkdir("PNAS")
        while os.path.exists(dirname):
            i += 1
            dirname = f"PNAS/{i}/"
        os.mkdir(f"PNAS/{i}")
        np.save(dirname + name, inverted)
        with open( dirname + name + ".pkl", 'wb') as handle:
            pickle.dump(params, handle)

