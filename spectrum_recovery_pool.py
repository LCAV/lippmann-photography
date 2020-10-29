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
# import jax.numpy as np
import numpy as np
import pickle
from scipy.optimize import fmin, nnls
import scipy as sp
import time


from lippmann2 import *
import display_spectral_data as dsd
from safe_save import *

c0 = 299792458
n0 = 1.5
c = c0 / n0


def generate_matrix_F(omegas, Z, margin=0.5):
    """Create up-samping matrix, from size defined by Z to length of omegas"""
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
    """Generate matrix that maps power spectrum to reflected complex wave"""
    lambdas = 2 * np.pi * c / omegas

    mi = np.min(omegas)
    ma = np.max(omegas)
    delta = np.abs(omegas[0] - omegas[1])

    if over_samples == 0:
        omegas_over = omegas
    else:
        omegas_over = np.arange(ma + over_samples * delta, mi - over_samples * delta, -delta)
    lambdas_over = 2 * np.pi * c / omegas_over

    A = fda.h(lambdas_over, lambdas, Z, r=r, mode=2, k0=k0)
    A0 = fda.h(lambdas_over, lambdas, Z, r=r, mode=3, k0=k0)

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
    """Solve (if possible non negative) least squares to find a spectrum...? """
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


def forward_model(omegas, spectrum, r, Z, k0):
    """Calculate complex reflected wave given (power) spectrum """
    F = generate_matrix_F(omegas, Z)
    A = generate_matrix_A(omegas, Z, r=r, k0=k0)
    B = A @ F
    return B @ spectrum


def spectrum_recovery(omegas, signal_measured, r=-1, Z=6E-6, k0=0, n_iter=200,
                      estimate_depth=True, previous_spectrum=None, stop_error=None):
    """Used in spectrum_recovery_data"""

    lambdas = 2 * np.pi * c / omegas

    if estimate_depth:
        Z = estimate_Z(lambdas, signal_measured)

    Z_lb = Z
    F = generate_matrix_F(omegas, Z)
    signal_est = np.ones(F.shape[1])
    if previous_spectrum is not None:
        if signal_est.shape == previous_spectrum.shape:
            signal_est = previous_spectrum
    signal_measured_sqrt = np.sqrt(signal_measured)
    error = np.inf

    print(f"starting {n_iter} iterations")
    for i in range(n_iter):

        complex_wave = forward_model(omegas, signal_est, r, Z, k0)
        error = np.linalg.norm(signal_measured_sqrt - np.abs(complex_wave))/np.linalg.norm(signal_measured_sqrt)
        if stop_error is not None:
            if stop_error > error:
                print(f"reached error {error}")
                break
        complex_wave *= signal_measured_sqrt / np.abs(complex_wave)

        if estimate_depth:
            Z, k0 = refine_Z_tau(i, lambdas, complex_wave, signal_measured, Z, r, k0, lb=Z_lb)

        signal_est = power_spectrum_from_complex_wave(omegas, complex_wave, r, Z, k0)
        min_signal = np.min(signal_est)
        if min_signal < 0:
            print(f"signal estimated to have negative values")

    return generate_matrix_F(omegas, Z) @ signal_est, Z, k0, error, signal_est


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

        # print(Z, k0)
    elif np.mod(i, 5) == 0:

        error = lambda x: np.sum((np.abs(f(x)) - np.sqrt(signal_measured)) ** 2)
        xopt = fmin(error, x0=[Z, k0], disp=False)
        Z, k0 = np.abs(xopt[0]), np.maximum(0, xopt[1])
        #        Z = xopt[0]

        # print(Z, k0)

    return Z, k0


def spectrum_recovery_data(lambdas_nu, spectrum_nu, N=200, r=0.2, visible=True, Z=8E-6, n_iter=200,
                           estimate_depth=False, previous_spectrum=None, stop_error=None):
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

    spectrum_est, Z_est, k0_est, error, spectrum_low_res = spectrum_recovery(
        omegas, spectrum, r=r, Z=Z, k0=k0, n_iter=n_iter, estimate_depth=estimate_depth,
        previous_spectrum=previous_spectrum, stop_error=stop_error)

    return spectrum_est, Z_est, k0_est, spectrum_low_res, error


if __name__ == '__main__':
    """reflective index r = ρeexp(jθ).
     For an interface between glass and mercury ρ = 0.71 and θ = −148 
     for an interface between glass and air ρ = 0.2 and θ = 0."""

    plt.close('all')

    params = {"N": 200,
              "Z": 3E-6,
              "k0": 3.7,
              "r":  0.7 * np.exp(1j * np.deg2rad(148)),
              "n_iter": 300,
              "visible": True,
              "downsampling": 25,
              "estimate_depth": False,
              "stop_error": 0.005,
              "c": c,
              "mask_purple": False}

    directory = 'Cubes/'

    for file in glob.glob(directory + '/*.dat'):
        name = file.split('/')[-1][:-4]
        print("\n")
        print("processing file", name)

        if name == "parrot":
            params["Z"] = 3.35e-6
            params["k0"] = 3.09
        if name == "saasfee":
            params["Z"] = 2.76e-6
            params["k0"] = 3.42

        data, wavelengths = dsd.load_specim_data(directory + name, ds=params["downsampling"], cut=True)

        if params["mask_purple"]:
            def sigmoid(x):
                return 1 / (1 + np.exp(-x))

            mask = np.linspace(3, 50, len(wavelengths))
            mask = sigmoid(mask)
            data *= mask[None, None, :]

        start = time.time()

        def spectrum_recovery_row(x):
            inverted_row = np.zeros((data.shape[1], params["N"]))
            inverted_low_res_row = []
            Z_row = np.zeros((data.shape[1]))
            k0_row = np.zeros((data.shape[1]))
            e_row = np.zeros((data.shape[1]))
            row_time = time.time()
            for y in range(data.shape[1]):
                pixel_time = time.time()
                print(f"processing ({x}, {y})")
                spectrum_est, Z_est, k0_est, spectrum_low_res, error = \
                    spectrum_recovery_data(wavelengths,
                                           data[x, y, :],
                                           N=params["N"],
                                           r=params["r"],
                                           visible=params["visible"],
                                           Z=params["Z"],
                                           n_iter=params["n_iter"],
                                           estimate_depth=params["estimate_depth"],
                                           previous_spectrum=(None if y == 0 else spectrum_low_res),
                                           stop_error=params["stop_error"])
                inverted_row[y, :] = spectrum_est
                inverted_low_res_row.append(spectrum_low_res)
                Z_row[y] = Z_est
                k0_row[y] = k0_est
                e_row[y] = error
                print(f"Pixel time: {time.time() - pixel_time}")
            print(f"Row time: {time.time() - row_time}")
            return inverted_row, Z_row, k0_row, e_row, np.array(inverted_low_res_row)


        pool = Pool(1)
        inverted, Z_estimates, k0_estimates, errors, inverted_low_res = zip(*pool.map(
            spectrum_recovery_row, range(data.shape[0])))
        inverted = np.array(inverted)
        inverted_low_res = np.array(inverted_low_res).astype(float)
        params["errors"] = np.array(errors)
        if params["estimate_depth"]:
            params["Z_estimates"] = np.array(Z_estimates)
            params["k0_estimates"] = np.array(k0_estimates)
        print(f"Time elapsed: {time.time() - start}")

        print(f"Min estimated: {np.min(inverted)}")

        i = 0
        dirname = f"PNAS/{i}/"
        if not os.path.exists("PNAS"):
            os.mkdir("PNAS")
        while os.path.exists(dirname):
            i += 1
            dirname = f"PNAS/{i}/"
        os.mkdir(f"PNAS/{i}")
        np.save(dirname + name, inverted)
        np.save(dirname + name + "_low_res", inverted_low_res)
        with open(dirname + name + ".pkl", 'wb') as handle:
            pickle.dump(params, handle)

