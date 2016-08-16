# -*- coding: utf-8 -*-
"""
Created on Sun Jul 03 2016

@author: Gilles Baechler
"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import numpy as np
from scipy import integrate

class Spectrum(object):
    
    def __init__(self, wave_lengths, intensities):
        
        self.wave_lengths = wave_lengths
        self.intensities = intensities
        
    def show(self, x=0, y=0, title='', sqrt=False):
        
        if sqrt:
            intensity = np.sqrt(np.abs(self.intensities[x,y,:]))
        else:
            intensity = self.intensities[x,y,:]
        
        plt.figure()
        l = len(self.wave_lengths)
        colors = plt.cm.Spectral_r(np.linspace(0, 1, l))
        cs = [colors[i] for i in range(l)]
        plt.scatter(self.wave_lengths, intensity, color=cs)
        plt.plot(self.wave_lengths, intensity, '--k', linewidth=1.0, zorder=-1, alpha=0.5)
        plt.gca().set_xlim([np.min(self.wave_lengths), np.max(self.wave_lengths)])        
        plt.gca().set_xlabel('Wavelength (m)')
        plt.title(title)
        plt.show()
        
    def __setitem__(self, key, value):        
        self.intensities[:, :, key] = value
        
    def __getitem__(self, key):        
        return self.intensities[:, :, key]
        
class Spectrogram(object):
    
    def __init__(self, coord, intensities):
        
        self.coord = coord
        self.intensities = intensities
        
    def show(self, title=''):
        plt.figure()
        plt.plot(self.coord[:,2], self.intensities)
        plt.gca().set_xlim([np.min(self.coord[:,2]), np.max(self.coord[:,2])])        
        plt.gca().set_xlabel('Depth (m)')
        plt.title(title)
        
class PlaneWave(object):
    
    def __init__(self, k, n=1.0, E_0=1.0, phi_0=np.pi/2.0):
        
        self.E_0 = E_0
        self.phi_0 = phi_0
        self.A = E_0*np.exp(1j*phi_0)             #complex envelope
        self.n = n

        #speed of light 
        self.c0 = 299792458
        self.c  = self.c0/n
        
        self.k = k                                #wavevector
        self.wavenumber = np.linalg.norm(self.k)  #wavenumber
        self.lambd = 2.0*np.pi/self.wavenumber    #wavelength
        self.omega = self.wavenumber*self.c       #angular freq
        
    def phase(self, r, sym=False):
        #sym returns the phase of the reflected wave
        return self.phi_0 + (2*sym-1)*r.dot(self.k)
        
    def complex_amplitude(self, r):
        return self.A*np.exp( -1j*r.dot(self.k) )
        
    def wave_function(self, r, t, real=True):
        if real:
            return np.real( self.complex_amplitude(r)*np.exp(1j*self.omega*t) )
        else:
            return self.complex_amplitude(r)*np.exp(1j*self.omega*t)
        
    def intensity(self):
        return self.A*np.conj(self.A)
        
    def standing_wave_intensity(self, r):
        return 2*self.intensity()*(1 - np.cos(self.phase(r, sym=True) - self.phase(r)) )
        
        
class PolychromaticPlaneWave(object):
    
    def __init__(self, direction, spectrum, n=1.0, E_0=1.0, phi_0=np.pi/2.0):
        
        self.E_0 = E_0
        self.phi_0 = phi_0
        self.A = spectrum.intensities*E_0*np.exp(1j*phi_0)       #complex envelope
        self.n = n
        
        self.spectrum = spectrum

        #speed of light 
        self.c0 = 299792458
        self.c  = self.c0/n
        self.epsilon_0 = 8.8541878176E-12
        
        self.direction = direction
        self.lambdas = spectrum.wave_lengths  #wavelength
        self.ks = 2.0*np.pi/self.lambdas      #wavevector
        self.omegas = self.ks*self.c          #angular freq
        
    def phases(self, r, sym=False):
        #sym returns the phase of the reflected wave
        return self.phi_0 + (2*sym-1)*r.dot(np.outer(self.direction, self.ks) ).T
        
    def complex_amplitude(self, r):
        return self.A[:, np.newaxis]*np.exp( -1j*r.dot(np.outer(self.direction, self.ks)) ).T
        
    def wave_function(self, r, t, real=True):
        
        U_r = self.complex_amplitude(r)
        U_t = np.exp(1j*np.outer(self.omegas, t))

        #numerical integration
        w_f = integrate.simps(y=(U_r[:,:,np.newaxis] * U_t[:,np.newaxis,:]), x=self.lambdas, axis=0)
        
        if real:
            return np.real(w_f)
        else:
            return w_f
        
    def intensity(self, r):
        U_r = self.complex_amplitude(r)
        #numerical integration
        return 2*self.c*self.epsilon_0*integrate.simps(y=(U_r*np.conj(U_r)), x=self.lambdas, axis=0)

class LippmannPlate(object):
    
    def __init__(self, direction=np.array([0.0, 0.0, 1.0]), wave_lengths, n_x, n_y, n=1.0, E_0=1.0, phi_0=np.pi/2.0):
        
        self.width  = n_x
        self.height = n_y
        self.spectrum = Spectrum(wave_lengths, [n_x, n_y, length(wave_lengths)]*None)        
        
        self.E_0 = E_0
        self.phi_0 = phi_0
        self.A = E_0*np.exp(1j*phi_0)       #complex envelope
        self.n = n

        #speed of light 
        self.c0 = 299792458
        self.c  = self.c0/n
        self.epsilon_0 = 8.8541878176E-12
        
        self.direction = direction
        self.ks = 2.0*np.pi/wave_lengths      #wavevector
        self.omegas = self.ks*self.c          #angular freq
        
    def phases(self, r, sym=False):
        #sym returns the phase of the reflected wave
        return self.phi_0 + (2*sym-1)*r.dot(np.outer(self.direction, self.ks) ).T
        
    def complex_amplitude(self, r):
        return self.A[:, np.newaxis]*np.exp( -1j*r.dot(np.outer(self.direction, self.ks)) ).T
        
    def wave_function(self, r, t, real=True):
        
        U_r = self.complex_amplitude(r)
        U_t = np.exp(1j*np.outer(self.omegas, t))

        #numerical integration
        w_f = integrate.simps(y=(U_r[:,:,np.newaxis] * U_t[:,np.newaxis,:]), x=spectrum.wave_lengths, axis=0)
        
        if real:
            return np.real(w_f)
        else:
            return w_f
        
    def intensity(self, r):
        U_r = self.complex_amplitude(r)
        #numerical integration
        return 2*self.c*self.epsilon_0*integrate.simps(y=(U_r*np.conj(U_r)), x=self.lambdas, axis=0)

    