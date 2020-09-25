#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 14:07:01 2018

@author: gbaechle
"""

import numpy as np
from scipy import misc
import matplotlib.pyplot as plt

from os import listdir
from os.path import isfile, join

import warnings

gdal_available = True
try:
    from osgeo import gdal
except ImportError as gdal_import_error:
    gdal_available = False
    warnings.warn(gdal_import_error.msg)

import sys

sys.path.append("../")

from spectrum import Spectrum3D
# from spectrum_recovery import *
from lippmann import *


# from hyperspectral_display import GuiManager


def read_file(path):
    f = open(path, 'r')
    lines = f.readlines()
    f.close()

    wavelengths = []
    data = []

    i_time = lines[6].split()[-1]

    for line in lines[14:]:
        s = line.split()
        wavelengths.append(float(s[0]))
        data.append(float(s[1]))

    return np.array(wavelengths) * 1E-9, np.array(data), float(i_time)


def normalize_data(wavelengths, data, i_time, white_path, black_path):
    _, white, i_white = read_file(white_path)
    _, black, i_black = read_file(black_path)

    #    return (data - black)/(white - black)
    return (data / i_time - black / i_black) / (white / i_time - black / i_black)


def read_ximea(path, white_ref='', plot=True):
    if white_ref != '':
        white = read_ximea(white_ref, plot=False)
        white = white.intensities[450, 1300, :]

    if not gdal_available:
        raise ImportError("to use PURDUE image module osgeo is required (need gdal.Open)")

    gtif = gdal.Open(path)

    # define wavelengths
    wavelengths = np.array([464, 476, 487, 498, 510, 522, 536, 546, 553, 565, 577, 589, 599, 608, 620, 630]) * 1E-9

    shape = gtif.GetRasterBand(1).GetDataset().ReadAsArray()[0].shape
    cube = np.zeros(shape + (len(wavelengths),))

    for idx in range(gtif.RasterCount - 1):
        print("[ GETTING BAND ]: ", idx)
        band = gtif.GetRasterBand(idx + 2)

        data = band.GetDataset().ReadAsArray()[idx + 1]
        cube[:, :, idx] = data

        if plot:
            plt.figure()
            plt.imshow(data, cmap='gray')

    if white_ref != '':
        cube /= white[None, None, :]
    #    cube /= np.max(cube)

    return Spectrum3D(wavelengths, cube)


def demosaic_ximea(path):
    bands = np.array([464, 476, 487, 498, 510, 522, 536, 546, 553, 565, 577, 589, 599, 608, 620, 630]) * 1E-9
    indices = [(0, 0), (1, 0), (2, 0), (3, 0), (3, 1), (2, 1), (1, 1), (0, 1), (3, 2), (2, 2), (1, 2), (0, 2), (3, 3),
               (2, 3), (1, 3), (0, 3)]

    img = misc.imread(path)

    plt.figure()
    plt.imshow(img, cmap='gray')

    cube = np.zeros((img.shape[0] // 4, img.shape[1] // 4, 16))

    for i, idx in enumerate(indices):
        cube[:, :, i] = img[idx[1]::4, idx[0]::4]
        plt.figure()
        plt.imshow(img[idx[1]::4, idx[0]::4], cmap='gray')

    return Spectrum3D(bands, cube)


def spectrum_rec(lambdas_nu, spectrum_nu, visible=True, N=200, r=0.7 * np.exp(np.deg2rad(-148))):
    omegas_nu = 2 * np.pi * c / lambdas_nu
    if visible:
        omegas = np.linspace(2 * np.pi * c / 400E-9, 2 * np.pi * c / 700E-9, N)
    else:
        omegas = np.linspace(np.max(omegas_nu), np.min(omegas_nu), N)
    lambdas = 2 * np.pi * c / omegas
    spectrum = sp.interpolate.interp1d(omegas_nu, spectrum_nu, kind='cubic', bounds_error=False,
                                       fill_value='extrapolate')(omegas)

    spectrum_est, Z_est, k0_est = spectrum_recovery(omegas, spectrum, r=r, Z=0, k0=0, n_iter=200, plot=False)

    depths = np.linspace(0, Z_est, N)

    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(3.45 / 0.6 * 1.5, 3.45 / 4 / 0.6 * 1.5))
    show_spectrum(lambdas, spectrum, ax=ax1, show_background=True, short_display=True)
    show_spectrum(lambdas, np.abs(np.real(fda.apply_h(spectrum_est, Z_est, lambdas, lambdas, r=r, k0=k0_est, mode=2))),
                  ax=ax2, show_background=True, short_display=True)
    show_lippmann_transform(depths, lippmann_transform(lambdas, spectrum_est, depths, r=r, k0=k0_est)[0], ax=ax3,
                            short_display=True)
    show_spectrum(lambdas, np.abs(fda.apply_h(spectrum_est, Z_est, lambdas, lambdas, r=r, k0=k0_est, mode=1)) ** 2,
                  ax=ax4, short_display=True, show_background=True)
    ax1.set_title('(a) Measured spectrum')
    ax2.set_title('(b) Estimated spectrum')
    ax3.set_title('(c) Estimated silver density')
    ax4.set_title('(d) Estimated replayed spectrum')
    #    f.tight_layout()
    #    f.subplots_adjust(hspace=-1.0)

    return spectrum_est, Z_est, k0_est


def dye_profile(path):
    files = sorted([f for f in listdir(path) if isfile(join(path, f))])

    wavelengths = []
    intensity = []

    # discard the last one (light reference), and the first one (noisy data)
    for file in sorted(files)[2:-1]:
        print(file)

        w, spectrum, _ = read_file(join(path, file))

        # select only visible light
        w, spectrum = select_visible(w, spectrum)

        show_spectrum(w, spectrum)

        peak_idx = np.argmax(spectrum)
        wavelengths.append(w[peak_idx])
        intensity.append(spectrum[peak_idx])

        print(w[peak_idx], spectrum[peak_idx])

    # remove the edge values that are not so reliable...
    wavelengths = np.array(wavelengths[:-7])
    intensity = np.array(intensity[:-7])

    show_spectrum(wavelengths, intensity)
    plt.title('Without correction')

    w, light, _ = read_file(join(path, files[-1]))
    w, light = select_visible(w, light)
    show_spectrum(w, light)
    plt.title('Light source')

    intensity /= np.interp(wavelengths, w, light)

    show_spectrum(wavelengths, intensity)
    plt.title('After correction')

    spread = np.abs(np.diff(wavelengths))

    print(intensity)

    spread[spread == 0] = 10E-10
    intensity = intensity[1:] / spread

    print(intensity)

    show_spectrum(wavelengths[1:], intensity)
    plt.title('After correction 2')


def select_visible(wavelengths, spectrum):
    idx = np.where((wavelengths <= 700E-9) & (wavelengths >= 402E-9))
    wavelengths = wavelengths[idx]
    spectrum = spectrum[idx]

    return wavelengths, spectrum


def load_specim_data(file_prefix, ds, cut=False):
    """Load binary data from .dat file and wavelengths from header .hdr file

    :param file_prefix: file name to read, without extension
    :param ds: down-sampling parameter (how many times to reduce image size)
    :param cut: if True, use data from file_prefix_cut.txt to cut edges of the image
    :returns:
        a pair of np. arrays, 2D cropped image and list of wavelengths (in meters)"""

    data = np.fromfile(file_prefix + ".dat", dtype=np.float32)
    data = np.swapaxes(data.reshape((512, -1, 512)), 1, 2)
    if cut:
        cut_idx = np.loadtxt(file_prefix + "_cut.txt").astype(np.int)
        data = data[cut_idx[0, 0]:cut_idx[0, 1], cut_idx[1, 0]:cut_idx[1, 1]]
    downsampled = data[::ds, ::ds, :]

    header_file = open(file_prefix + ".hdr", "r")
    header = header_file.read()
    wavelengths = np.array([float(nr[:-1]) for nr in header.split("{")[-1].split()[:-1]])

    return downsampled, 1e-9 * wavelengths


# if __name__ == '__main__':
    # path = '/Users/gbaechle/EPFL/PhD/BRDF Data and Rendering Engines/Spectrometer/2018-09-18 Wiener'

    # dye_profile(path)

#    path =       '/Users/gbaechle/EPFL/PhD/BRDF Data and Rendering Engines/Lippmann Elysée/spectrometer_data/Parrot 1.txt'
#    
#    wavelengths, data, i_time = read_file(path)
#        
#    print('integration time', i_time)
#    
#    white_path = '/Users/gbaechle/EPFL/PhD/BRDF Data and Rendering Engines/Lippmann Elysée/spectrometer_data/Bright frame ' + str(int(i_time)) + '.txt'
#    black_path = '/Users/gbaechle/EPFL/PhD/BRDF Data and Rendering Engines/Lippmann Elysée/spectrometer_data/Dark frame ' + str(int(i_time)) + '.txt'
#    
#    
#    show_spectrum(wavelengths, data, visible=True, show_background=True)
#    
#    data_norm = normalize_data(wavelengths, data, i_time, white_path, black_path)
#    show_spectrum(wavelengths, data_norm, visible=True, show_background=True)

#    spectrum_est, Z_est, k0_est = spectrum_rec(wavelengths, data_norm)
#    show_spectrum(wavelengths, data_norm, visible=True, show_background=True)

#    wavelengths, data, i_time = read_file(white_path)
#    show_spectrum(wavelengths, data, visible=True)
#    wavelengths, data, i_time = read_file(black_path)
#    show_spectrum(wavelengths, data, visible=True)


#    for i_time in [10, 15, 25, 30, 40, 60]:
#        white_path = '/Users/gbaechle/EPFL/PhD/BRDF Data and Rendering Engines/Lippmann Elysée/spectrometer_data/Bright frame ' + str(int(i_time)) + '.txt'
#        black_path = '/Users/gbaechle/EPFL/PhD/BRDF Data and Rendering Engines/Lippmann Elysée/spectrometer_data/Dark frame ' + str(int(i_time)) + '.txt'
#    
#        wavelengths, data_w, i_time = read_file(white_path)
#        wavelengths, data_b, i_time = read_file(black_path)
#        show_spectrum(wavelengths, data_w-data_b, visible=True, vmax=38837)
#        plt.gca().set_title('white' + str(i_time))
#        show_spectrum(wavelengths, data_b, visible=True, vmax=20582)
#        plt.gca().set_title('dark' + str(i_time))


#    path = '/Users/gbaechle/EPFL/PhD/BRDF Data and Rendering Engines/Lippmann Elysée/Ximea/colorchecker_2_8.tif'
#    spectrum = demosaic_ximea(path)
#    print(spectrum.intensities.shape)
#    spectrum.compute_rgb()
#    print(spectrum.intensities.shape, spectrum.rgb_colors.shape)

#    path = '/Volumes/Gilles EPFL/Datasets/Ximea hyperspectral/Demixed_parrot_200_8.tif'
#    path_white = '/Volumes/Gilles EPFL/Datasets/Ximea hyperspectral/Demixed_white_50_8.tif'
#    spectrum = read_ximea(path, white_ref=path_white)
#    spectrum.intensities = np.transpose(spectrum.intensities, (2,0,1))
#    spectrum.compute_rgb(integrate_nu=True)
#    spectrum.intensities = np.transpose(spectrum.intensities, (1,2,0))
#    
#    gui_manager = GuiManager(spectrum, normalize_spectrums=True, gamma_correct=False)
#    gui_manager.show()
