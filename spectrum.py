# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 13:55:23 2016

@author: gbaechle
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

import color_tools as ct


class Spectrum(object):
    def __init__(self, wave_lengths, intensities):

        self.wave_lengths = wave_lengths
        self.intensities = intensities

    def show(self, ax=None, title='', sqrt=False, y_max=0.):

        if sqrt:
            intensity = np.sqrt(np.abs(self.intensities))
            y_max = np.sqrt(y_max)
        else:
            intensity = self.intensities

        if ax is None:
            plt.figure()
            ax = plt.gca()

        ax.clear()

        # sort the wavelengths in order
        idx = np.argsort(self.wave_lengths)
        wl = self.wave_lengths[idx]
        intensity = intensity[idx]

        # select only the visible spectrum
        idx2 = np.where((wl >= 390E-9) & (wl <= 700E-9))
        wl = wl[idx2]
        intensity = intensity[idx2]

        l = len(wl)

        colors = plt.cm.Spectral_r(np.linspace(0, 1, l))
        cs = [colors[i] for i in range(l)]
        ax.scatter(wl, intensity, color=cs)
        ax.plot(wl, intensity, '--k', linewidth=1.0, zorder=-1, alpha=0.5)
        ax.set_xlim([np.min(wl), np.max(wl)])
        if y_max != 0.:
            ax.set_ylim([0., y_max])
        ax.set_xlabel('Wavelength (m)')
        ax.set_title(title)


class Spectrum3D(object):
    def __init__(self, wave_lengths, intensities):

        self.wave_lengths = wave_lengths
        self.intensities = intensities

        self.xyz_colors = None
        self.rgb_colors = None

    def show(self, x=0, y=0, title='', sqrt=False):

        if sqrt:
            intensity = np.sqrt(np.abs(self.intensities[x, y, :]))
        else:
            intensity = self.intensities[x, y, :]

        plt.figure()

        idx = np.argsort(self.wave_lengths)
        wl = self.wave_lengths[idx]
        intensity = intensity[idx]

        l = len(self.wave_lengths)

        colors = plt.cm.Spectral_r(np.linspace(0, 1, l))
        cs = [colors[i] for i in range(l)]
        plt.scatter(wl, intensity, color=cs)
        plt.plot(wl, intensity, '--k', linewidth=1.0, zorder=-1, alpha=0.5)
        plt.gca().set_xlim([np.min(wl), np.max(wl)])
        plt.gca().set_xlabel('Wavelength (m)')
        plt.title(title)
        plt.show()

    def compute_xyz(self, sqrt=False):

        if self.xyz_colors is None:
            idx = np.argsort(self.wave_lengths)
            if sqrt:
                self.xyz_colors = ct.from_spectrum_to_xyz(self.wave_lengths[idx], np.sqrt(self.intensities[:, :, idx]),
                                                          integrate_nu=False)
            else:
                self.xyz_colors = ct.from_spectrum_to_xyz(self.wave_lengths[idx], self.intensities[:, :, idx],
                                                          integrate_nu=False)

        return self.xyz_colors

    def compute_rgb(self, sqrt=False):

        if self.rgb_colors is None:

            if self.xyz_colors is None:
                self.compute_xyz(sqrt)

            self.rgb_colors = ct.from_xyz_to_rgb(self.xyz_colors)

        return self.rgb_colors

    def blue_shift(self, factor, extrapolation='zero'):

        f = interp1d(self.wave_lengths / factor, self.intensities, axis=2, kind='cubic', bounds_error=False,
                     fill_value=0.)
        #        f = interp1d(self.wave_lengths*np.cos(factor), self.intensities, axis=2, kind='cubic', bounds_error=False, fill_value=0.)

        max_vals = self.intensities[:, :, 0]
        self.intensities = np.maximum(f(self.wave_lengths), 0.)
        max_wavelength = np.max(self.wave_lengths)

        if extrapolation == 'cste':
            idx = np.where(self.wave_lengths * factor > max_wavelength)[0]
            self.intensities[:, :, idx] = max_vals[:, :, np.newaxis]

    def __setitem__(self, key, value):
        self.intensities[:, :, key] = value

    def __getitem__(self, key):
        return self.intensities[:, :, key]

    def get_spectrum(self, x, y):
        return Spectrum(self.wave_lengths, self.intensities[x, y, :])

    def set_spectrum(self, x, y, value):
        self.intensities[x, y, :] = value


class Scale(object):
    def __init__(self, depths, intensities):
        self.depths = depths
        self.intensities = intensities

    def show(self, title=''):
        plt.figure()
        plt.plot(self.depths, self.intensities)
        plt.gca().set_xlim([np.min(self.depths), np.max(self.depths)])
        plt.gca().set_xlabel('Depth (m)')
        plt.title(title)
