# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 11:19:49 2016

@author: Gilles Baechler
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
from mpl_toolkits.mplot3d import Axes3D
from scipy import integrate
from scipy.interpolate import interp1d
from scipy.fftpack import dct

from spectrum import *



class Lippmann(object):
    """Class defining a generic Lippmann object"""
    
    def __init__(self, spectrum, n_x, n_y, r, direction=np.array([0.0, 0.0, 1.0]), light_spectrum=None, spectral_sensitivity=None, n=1.0, phi_0=np.pi/2.0, E_0=1):
        
        self.spectrum = spectrum
        self.wavelengths = spectrum.wave_lengths
        l = len(self.wavelengths)
        
        self._I = None
        self._R = None
        self._I_r = None
        
        #Physical constants
        self.c0 = 299792458
        self.c  = self.c0/n
        self.epsilon_0 = 8.8541878176E-12        

        self.width  = n_x
        self.height = n_y
    
        self.r = r
        
        self.E_0 = E_0
        self.phi_0 = phi_0
        self.A = E_0*np.exp(1j*phi_0)       #complex envelope
        self.n = n
        
        self.direction = direction/np.linalg.norm(direction)  #make sure it is a unit vector
        self.k         = 2*np.pi/self.wavelengths                 #wavenumber
        self.k_vec     = self.k[None,:]*direction[:,None]     #wavevector
        self.omega     = self.k*self.c                        #angular freq
        self.nu        = self.omega/(2*np.pi)
        
        #If the incoming light spectrum is not provided, assume uniform light
        if light_spectrum is None:
            self.light_spectrum = Spectrum( self.wavelengths, np.ones(l) )
        else:
            self.light_spectrum = light_spectrum
    
        #If the spectral sensitivity is not provided, assume uniform spectral sensitivity
        if spectral_sensitivity is None:
            self.spectral_sensitivity = Spectrum( self.wavelengths, np.ones(l) )
        else:
            self.spectral_sensitivity = spectral_sensitivity  
            
        self.plate_type = 'continuous'
            
    def phases(self, r=None, sym=False):
        #returns the phase of the reflected wave
        if r is None:
            r = self.r
        return self.phi_0 + (2*sym-1)*r @ self.k_vec
        
    def complex_amplitude(self, r=None):
        if r is None:
            r = self.r
        return self.A*np.exp( -1j*r @ self.k_vec)
        
        
    def compute_wave_function(self, t, r=None, real=True):
        """Compute the incoming wave function."""        
        
        U_r = self.complex_amplitude(r)
        U_t = np.exp(1j*np.outer(self.omega, t))
        
        print(U_r.shape)
        print(U_t.shape)

        #numerical integration
        w_f = np.trapz(y=(U_r[:,:,None] * U_t[None,:,:]), x=self.wavelengths, axis=1)
        
        if real:
            return np.real(w_f)
        else:
            return w_f
  
        
    def compute_intensity(self):
        """Compute the intensity of the interference field as well as the reflectivity of the Lippmann plate"""
        raise NotImplementedError( "This function should be implemented" )
        
    def replay(self):
        """Compute the corresponding spectrum reflected by the Lippmann plate"""
        raise NotImplementedError( "This function should be implemented" )
            

    @property
    def I(self):
        if self._I is None:
            self.compute_intensity()
        return self._I
        
    @I.setter
    def I(self, value):
        self._I = value
        
    @property
    def R(self):
        if self._R is None:
            self.compute_intensity()
        return self._R
    
    @R.setter
    def R(self, value):
        self._R = value
        
    @property
    def I_r(self):
        if self._I_r is None:
            self.replay()
        return self._I_r
    
    @I_r.setter
    def I_r(self, value):
        self._I_r = value
        
    def __str__(self):
        return self.plate_type + ' Lippmann plate of size (' + str(self.width) + ', ' + str(self.height) + ')' 
    
        

class LippmannContinuous(Lippmann):
    """Class defining a 'continuous' Lippmann object"""
    
    def __init__(self, wavelengths, n_x, n_y, r=None, direction=np.array([0.0, 0.0, 1.0]), light_spectrum=None, spectral_sensitivity=None, n=1.0, E_0=1.0, phi_0=np.pi/2.0):

        l = len(wavelengths)
        spectrum = Spectrum3D(wavelengths, np.zeros([n_x, n_y, l]))
        
        #default depth discretization
        if r is None:
            n_space = 1000
            self.r = np.zeros([n_space, 3])
            self.r[:,2] = np.linspace(0, 100.0E-6, n_space)
        else:
            self.r = r
                
        super().__init__(spectrum, n_x, n_y, r=self.r, direction=direction, light_spectrum=light_spectrum, spectral_sensitivity=spectral_sensitivity, n=n, phi_0=phi_0)
        
        self.plate_type = 'continuous'        
        
        
    def compute_intensity(self):
            
        self._I   = np.zeros([self.width, self.height, self.r.shape[0]])
        self._R   = np.zeros([self.width, self.height, self.r.shape[0]])
        
        kTr       = self.r @ self.k_vec
        sines     = 0.5*(1 - np.cos(2.0*self.n*kTr/self.direction[2]))/self.direction[2]
        
        i = self.light_spectrum.intensities[:, None] * sines.T
        r = self.spectral_sensitivity.intensities[:, None] * \
            self.light_spectrum.intensities[:, None] * sines.T
        
        for x in range(self.width):
            perc = np.double(x)/np.double(self.width-1)*100
            sys.stdout.write("\rComputing intensity: %.2f %%" %perc)
            sys.stdout.flush()                
            
            for y in range(self.height):
                integrand_i = self.spectrum.intensities[x, y, :, None] * i
                integrand_r = self.spectrum.intensities[x, y, :, None] * r
    
#                self._I[x, y,:] = np.trapz(y=integrand_i, x=self.nu, axis=0)
#                self._R[x, y,:] = np.trapz(y=integrand_r, x=self.nu, axis=0)
                self._I[x, y,:] = np.trapz(y=integrand_i, axis=0)
                self._R[x, y,:] = np.trapz(y=integrand_r, axis=0)
                
                
        sys.stdout.write("\nIntensity computed!\n")
        
        
    def replay(self, wavelengths=None, direction=np.array([0.0, 0.0, 1.0])):
        
        if self._I_r is None:
            
            if self._R is None:
                self.compute_intensity()            
            
            if wavelengths is None:
                wavelengths=self.wavelengths

            direction = direction / np.linalg.norm(direction)
                
            self._I_r = Spectrum3D(wavelengths, np.zeros([self.width, self.height, len(wavelengths)]))
   
            kTr     = self.r @ self.k_vec
            cosines = np.cos(2*self.n*kTr / direction[2]).T /direction[2]
                            
            for x in range(self.width):
                perc = x/(self.width-1)*100
                sys.stdout.write("\rComputing new spectrum: %.2f %%" %perc)
                sys.stdout.flush() 
                
                for y in range(self.height):
                    
                    integrand = self._R[x, y, :, np.newaxis] * cosines.T
                    
#                    self.new_spectrums.set_spectrum( x, y, 0.5*self.c*self.epsilon_0*self.light_spectrum.intensities * np.trapz(y=integrand, x=self.r[:,2], axis=0)**2 )
                    self._I_r.set_spectrum( x, y, self.light_spectrum.intensities * np.trapz(y=integrand, x=self.r[:,2], axis=0)**2 )
            
            sys.stdout.write("\nNew spectrum computed!\n")
            
        return self._I_r
        
        
    def to_uniform_freq(self, N_prime):
    
        lambda_max = np.max(self.wavelengths)
        lippmann_discrete = LippmannDiscrete(N_prime, self.width, self.height, lambda_max=lambda_max,  light_spectrum=self.light_spectrum, \
                            spectral_sensitivity=self.spectral_sensitivity, n=self.n, E_0=self.E_0, phi_0=self.phi_0)
                            
        old_wave_lengths = self.wavelengths
        new_wave_lengths = lippmann_discrete.spectrum.wave_lengths

        #interpolate
        f1 = interp1d(old_wave_lengths, self.spectrum.intensities, axis=2, fill_value='extrapolate')
        f2 = interp1d(old_wave_lengths, self.light_spectrum.intensities, fill_value='extrapolate')
        f3 = interp1d(old_wave_lengths, self.spectral_sensitivity.intensities, fill_value='extrapolate')
        lippmann_discrete.spectrum.intensities = f1(new_wave_lengths)
        lippmann_discrete.light_spectrum.intensities = f2(new_wave_lengths)
        lippmann_discrete.spectral_sensitivity.intensities = f3(new_wave_lengths)
        
        return lippmann_discrete
          
    
        
class LippmannDiscrete(Lippmann):
    """Class defining a discrete Lippmann object"""
    
    def __init__(self, N_prime, n_x, n_y, lambda_min=390E-9, lambda_max=700E-9, direction=np.array([0.0, 0.0, 1.0]), light_spectrum=None, spectral_sensitivity=None, n=1.0, E_0=1.0, phi_0=np.pi/2.0):
        
        self.c = 299792458      
        
        #reds (or higher)
        v_min = self.c/lambda_max
        #blues
        v_max = self.c/lambda_min      
        
        self.N_prime = N_prime
        self.N = np.int( np.floor(N_prime*v_max/(v_max-v_min)) ) 
        
        self.f_max = 2./(lambda_min)
        self.df    = self.f_max/(self.N-1)
        self.dr    = 1./(2.*self.f_max)         
        
        self.z           = np.arange(self.N)*self.dr
        self.r           = np.zeros([self.N, 3])
        self.r[:,2]      = self.z
        
        self.f           = np.arange(self.N)*self.df
        self.wavelengths     = 2./self.f[-self.N_prime:]

        spectrum = Spectrum3D(self.wavelengths, np.zeros([n_x, n_y, N_prime]))

        super().__init__(spectrum, n_x, n_y, r=self.r, direction=direction, light_spectrum=light_spectrum, spectral_sensitivity=spectral_sensitivity, n=n, phi_0=phi_0)

        self.plate_type = 'discrete'  
        
        
    def compute_intensity(self):
        
        if self._I is None:
            
            self._I = np.zeros([self.width, self.height, self.N])
            self._R = np.zeros([self.width, self.height, self.N])
                        
            for x in range(self.width):
                perc = np.double(x)/np.double(self.width-1)*100
                sys.stdout.write("\rComputing intensity: %.2f %%" %perc)
                sys.stdout.flush() 
                
                for y in range(self.height):
                    
                    #pad zeros to account for the lowpass band
                    x_i = np.hstack( [ np.zeros(self.N-self.N_prime), self.spectrum.intensities[x, y, :]*self.light_spectrum.intensities ] )
                    x_r = np.hstack( [ np.zeros(self.N-self.N_prime), self.spectrum.intensities[x, y, :]*self.spectral_sensitivity.intensities*self.light_spectrum.intensities ] )

                    g_i = dct(x=x_i, type=1)
                    g_r = dct(x=x_r, type=1)

                    self._I[x, y,:]  = g_i[0] - g_i
                    self._R[x, y,:] = g_r[0] - g_r

                    
    def replay(self):
                
        if self._I_r is None:
            
            if self._R is None:
                self.compute_intensity()           
                
            self._I_r = Spectrum3D(self.wavelengths, np.zeros([self.width, self.height, self.N_prime]))
   
   
            for x in range(self.width):
                perc = np.double(x)/np.double(self.width-1)*100
                sys.stdout.write("\rComputing new spectrum: %.2f %%" %perc)
                sys.stdout.flush()
                
                for y in range(self.height):
                    
                    #Normalize    
                    G = 1./(2*(self.N-1))*dct(self._R[x,y,:], type=1)
                    self._I_r.set_spectrum( x, y, self.light_spectrum.intensities*G[-self.N_prime:]**2 )
            
     
    



        
        
    
    