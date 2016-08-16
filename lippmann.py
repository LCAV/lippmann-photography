# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 11:19:49 2016

@author: Gilles Baechler
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from scipy import integrate
from scipy.interpolate import interp1d
from scipy.fftpack import dct
from itertools import product

import color_tools as ct
from spectrum import *


class LippmannPixel(object):
    """Class defining a Lippmann object for a single pixel (no spatial variation)"""
    
    def __init__(self, object_spectrum, n = 1.0, E0 = 1.0, light_spectrum=None, spectral_sensitivity=None):
        """Creates a new Lippmann pixel
        
        Args:
            object_spectrum:       the spectrum of the polychromatic wave of the object
             n (double):           refraction index (default: 1.0)
             E0 (double):          baseline amplitude (default: 1.0)
             light_spectrum:       the spectrum of the light illuminating the object (default: None, which is equivalent to all ones)
             spectral_sensitivity: the sensitivity of the plate to each frequency (default: None, which is equivalent to all ones)
            
        Returns:
            A Lippmann pixel object
        """     
        
        self.n = n
        self.E0 = E0
        self.object_spectrum = object_spectrum
        
        l = len(object_spectrum.intensities)
        
        if light_spectrum is None:
            self.light_spectrum = Spectrum( object_spectrum.wave_lengths, np.ones(l) )
        else:
            self.light_spectrum = light_spectrum
    
        if spectral_sensitivity is None:
            self.spectral_sensitivity = Spectrum( object_spectrum.wave_lengths, np.ones(l) )
        else:
            self.spectral_sensitivity = spectral_sensitivity
            
        self.E            = None
        self.reflectivity = None
        self.intensity    = None
        
        #speed of wave front
        self.C = 299792458/n
        self.epsilon_0 = 8.8541878176E-12
        
        
    def get_electric_field(self,t,z,display=False):
        """Compute the electric field at time t and depth z
        
        Args:
            t:              time as an nparray
            z:              depth as an nparray
            display (bool): whether or not to display it (default: False)
            
        Returns:
            The corresponding 2D electric field in a 2D nparray
        """

        if self.E is None:

            t_grid, z_grid = np.meshgrid(t,z)
            lambdas = self.object_spectrum.wave_lengths  
            nu      = self.C/lambdas
    
            self.E = np.zeros(t_grid.shape)
            
            y = np.zeros((len(lambdas), t_grid.shape[0], t_grid.shape[1]))
            
            for idx, lambd in enumerate(lambdas):
                #freq = self.C/lambd
                ang_wavenumber = 2*np.pi/lambd
                
                y[idx, :, :] = np.sqrt( self.light_spectrum.intensities[idx]*self.object_spectrum.intensities[idx] ) * \
                               np.sin(self.n*z_grid*ang_wavenumber)*np.sin(t_grid*self.C*ang_wavenumber)
            
            #numerical integration
            self.E = 2*self.E0 * np.trapz(y=y, x=nu, axis=0)
            
        if display:
            self.show_electric_field(t,z)
         
        return self.E
            
    def get_intensity(self, z, integration_method='trapz'):
        """Compute the electric field at depth z
        
        Args:
            z:                  depth as an nparray
            integration_method: can be either 'trapz', 'sum', or 'simps' (default: 'trapz')
            
        Returns:
            The corresponding intensity as a nparray
        """
        
        if not self.intensity is None:
            return self.intensity

        lambdas = self.object_spectrum.wave_lengths
        nus     = self.C/lambdas
        
        y = np.zeros((len(lambdas), len(z)))
        
        for idx, lambd in enumerate(lambdas):
#            sines = np.sin(2*np.pi*self.n*z/lambd)**2
            sines = 1.0 -np.cos(4*np.pi*self.n*z/lambd)
            
            y[idx, :] = (self.light_spectrum.intensities[idx]*self.object_spectrum.intensities[idx])*\
                        sines

        #numerical integration         
#        I = self.E0**2*self.C*self.n*self.epsilon_0 * np.trapz(y=y, x=lambdas, axis=0)
        
        if integration_method == 'trapz':
            I = self.E0**2*self.n * np.trapz(y=y, x=nus, axis=0)
        elif integration_method == 'sum':
            I = np.sum(y, axis=0)/len(nus)
        else:
            I = self.E0**2*self.n * integrate.simps(y=y, x=nus, axis=0)
        
        self.intensity = Spectrogram(z, I)
        return self.intensity
        
    def get_reflectivity(self, z, integration_method='trapz'):
        """Compute the reflectivity function of the plate at depth z
        
        Args:
            z:                  depth as an nparray
            integration_method: can be either 'trapz', 'sum', or 'simps' (default: 'trapz')
            
        Returns:
            The corresponding reflectivity as a nparray
        """
        
        if not self.reflectivity is None:
            return self.reflectivity
        
        lambdas = self.object_spectrum.wave_lengths
        nus     = self.C/lambdas        
        
        y = np.zeros((len(lambdas), len(z)))
        
        for idx, lambd in enumerate(lambdas):     
            sines = np.sin(2*np.pi*self.n*z/lambd)**2
#            sines = 0.5*(-np.cos(4*np.pi*self.n*z/lambd) )
            y[idx, :] = (self.light_spectrum.intensities[idx]*self.object_spectrum.intensities[idx]*self.spectral_sensitivity.intensities[idx]) * \
                        sines
                        
#            y[idx, :] = -(self.light_spectrum.intensities[idx]*self.object_spectrum.intensities[idx]*self.spectral_sensitivity.intensities[idx]) * \
#                        np.cos(4*np.pi*self.n*z/lambd)
            
        
        if integration_method == 'trapz':
            R = self.E0**2*self.n*self.epsilon_0 * np.trapz(y, x=nus, axis=0)
        elif integration_method == 'sum':
            R = np.sum(y, axis=0)/len(nus)
        else:
            R = self.E0**2*self.n*self.epsilon_0 * integrate.simps(y, x=nus, axis=0)
        
        
        self.reflectivity = Spectrogram(z,R)
        return self.reflectivity
        
    def re_illuminate(self, t, new_lambdas=None, new_light_spectrum=None, integration_method='trapz'):
        """Compute the spectrum back from the reflectivity function of the plate
        
        Args:
            t (double):         time (at which we evaluate the electric field)
            new_lambdas:        frequencies of evaluation of the spectrum (default: None, which reuses the one from the input wave)
            light_spectrum:     the spectrum of the light illuminating the plate (default: None, which is equivalent to all ones)            
            integration_method: can be either 'trapz', 'sum', or 'simps' (default: 'simps')
            
        Returns:
            The spectrum of the electric field and of the intensity
        """
        
#        epsilon_0 = 20E22
        epsilon_0 = 8.8541878176E-12
        epsilon_0 = 1.
        depths    = self.reflectivity.depths
        
        if new_lambdas is None:
            new_lambdas = self.object_spectrum.wave_lengths
        if new_light_spectrum is None:
            new_light_spectrum = self.light_spectrum
            
        if not self.reflectivity is None:
            self.get_reflectivity(depths)
        
        y1 = np.zeros((len(depths), len(new_lambdas)))
        y2 = np.zeros((len(depths), len(new_lambdas)))
        
        freqs = self.C/new_lambdas
        for idx, z in enumerate(depths):   
            y1[idx, :] = np.sqrt(new_light_spectrum.intensities)*self.reflectivity.intensities[idx]*np.sin( 4*np.pi*freqs*t - 4*np.pi*self.n/new_lambdas*z ) 
            y2[idx, :] = self.reflectivity.intensities[idx]*np.cos(4*np.pi*self.n/new_lambdas*z )
        
        if integration_method == 'trapz':
            E_new = self.E0*np.trapz(y=y1, x=depths, axis=0)
            I_new = self.n*self.C*epsilon_0/2.0*new_light_spectrum.intensities*self.E0**2* \
                    np.trapz(y=y2, x=depths, axis=0)**2
        elif integration_method == 'sum':
            E_new = np.sum(y1, axis=0)
            I_new = ( np.sum(y2, axis=0)/(2.*np.pi) )**2
        else:
            E_new = self.E0*integrate.simps(y=y1, x=depths, axis=0)
            I_new = self.n*self.C*epsilon_0/2.0*new_light_spectrum.intensities*self.E0**2* \
                    integrate.simps(y=y2, x=depths, axis=0)**2
        
        return Spectrum(new_lambdas, E_new), Spectrum(new_lambdas, I_new)
        
   
    def show_electric_field(self,t,z):
        """Display the elctric field
        
        Args:
            t: time as an nparray
            z: depth as an nparray
        """
               
        fig, ax = plt.subplots()
        ax.set_ylim([np.min(self.E),np.max(self.E)])
        self.line, = ax.plot(z, self.E[:,0])

        ani = animation.FuncAnimation(fig, self.animate, t.shape[0], interval=25, blit=False)
        ani.save('electric_field.mp4')      
        
        plt.show()
        
    def animate(self, t_idx):
        #print t_idx
        self.line.set_ydata(self.E[:,t_idx])
        return self.line
        
    
       
class LippmannPlate(object):
    
    def __init__(self, wave_lengths, n_x, n_y, r=None, direction=np.array([0.0, 0.0, 1.0]), light_spectrum=None, spectral_sensitivity=None, n=1.0, E_0=1.0, phi_0=np.pi/2.0):
        
        l = len(wave_lengths)        
        
        if light_spectrum is None:
            self.light_spectrum = Spectrum( wave_lengths, np.ones(l) )
        else:
            self.light_spectrum = light_spectrum
    
        if spectral_sensitivity is None:
            self.spectral_sensitivity = Spectrum( wave_lengths, np.ones(l) )
        else:
            self.spectral_sensitivity = spectral_sensitivity   
            
        if r is None:
            n_space = 2500
            self.r = np.zeros([n_space, 3])
            self.r[:,2] = np.linspace(0, 250.0E-6, n_space)
        else:
            self.r = r
        
        self.width  = n_x
        self.height = n_y
        self.spectrums = Spectrums(wave_lengths, np.zeros([n_x, n_y, l]))      
        
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
        
        self.intensities   = None
        self.reflectances  = None
        self.new_spectrums = None
        
    def getpixel(self, x, y):
        return LippmannPixel(n=1.0, object_spectrum=Spectrum(self.spectrums.wave_lengths, self.spectrums.intensities[x,y,:]))
    
    def phases(self, r, sym=False):
        #sym returns the phase of the reflected wave
        return self.phi_0 + (2*sym-1)*r.dot(np.outer(self.direction, self.ks) ).T
        
    def complex_amplitude(self, r):
        return self.A[:, np.newaxis]*np.exp( -1j*r.dot(np.outer(self.direction, self.ks)) ).T
        
    def wave_function(self, r, t, real=True):
        
        U_r = self.complex_amplitude(r)
        U_t = np.exp(1j*np.outer(self.omegas, t))

        #numerical integration
        w_f = np.trapz(y=(U_r[:,:,np.newaxis] * U_t[:,np.newaxis,:]), x=self.spectrums.wave_lengths, axis=0)
        
        if real:
            return np.real(w_f)
        else:
            return w_f
        
    def compute_intensity(self):
        
        if self.intensities is None:
            
            self.intensities    = np.zeros([self.width, self.height, self.r.shape[0]])
            self.reflectances   = np.zeros([self.width, self.height, self.r.shape[0]])
            
            kTr      = np.outer(self.ks, self.direction).dot(self.r.T)
#            sines     = np.sin(self.n*kTr)**2 
            sines     = 0.5*(1 - np.cos(2.0*self.n*kTr) )
#            sines     = -0.5*(np.cos(2.0*self.n*kTr) )
            
            for x in xrange(self.width):
                print x
                for y in xrange(self.height):
                    integrand_i = self.spectrums.intensities[x, y, :, np.newaxis] * \
                    self.light_spectrum.intensities[:, np.newaxis] * sines
                    
                    integrand_r = self.spectrums.intensities[x, y, :, np.newaxis] * \
                    self.spectral_sensitivity.intensities[:, np.newaxis] * \
                    self.light_spectrum.intensities[:, np.newaxis] * sines
                       
#                    self.intensities[x, y,:]  = 2*self.c*self.epsilon_0*np.trapz(y=integrand_i, x=self.spectrums.wave_lengths, axis=0)
#                    self.reflectances[x, y,:] = 2*self.c*self.epsilon_0*np.trapz(y=integrand_r, x=self.spectrums.wave_lengths, axis=0)
                    self.intensities[x, y,:]  = np.trapz(y=integrand_i, x=self.spectrums.wave_lengths, axis=0)
                    self.reflectances[x, y,:] = np.trapz(y=integrand_r, x=self.spectrums.wave_lengths, axis=0)
            
    
    def compute_new_spectrum(self, wave_lengths=None):
        
        if self.new_spectrums is None:
            
            if self.reflectances is None:
                self.compute_intensity()            
            
            if wave_lengths is None:
                wave_lengths=self.spectrums.wave_lengths
                
            self.new_spectrums = Spectrums( wave_lengths, np.zeros([self.width, self.height, len(wave_lengths)]) )
   
            kTr     = np.outer(self.ks, self.direction).dot(self.r.T)
            cosines = np.cos(2*self.n*kTr)
                            
            for x in xrange(self.width):
                print x
                for y in xrange(self.height):
                    
                    integrand = self.reflectances[x, y, :, np.newaxis] * cosines.T
                    
#                    self.new_spectrums.set_spectrum( x, y, 0.5*self.c*self.epsilon_0*self.light_spectrum.intensities * np.trapz(y=integrand, x=self.r[:,2], axis=0)**2 )
                    self.new_spectrums.set_spectrum( x, y, 1E20*self.light_spectrum.intensities * np.trapz(y=integrand, x=self.r[:,2], axis=0)**2 )
            
            
    def to_uniform_freq(self, N_prime):
    
        lippmann_discrete = LippmannPlateDiscrete(N_prime, self.width, self.height, light_spectrum=self.light_spectrum, \
                            spectral_sensitivity=self.spectral_sensitivity, n=self.n, E_0=self.E_0, phi_0=self.phi_0)
                            
        old_wave_lengths = self.spectrums.wave_lengths
        new_wave_lengths = lippmann_discrete.spectrums.wave_lengths

        #interpolate
        f1 = interp1d(old_wave_lengths, self.spectrums.intensities, axis=2, fill_value='extrapolate')
        f2 = interp1d(old_wave_lengths, self.light_spectrum.intensities, fill_value='extrapolate')
        f3 = interp1d(old_wave_lengths, self.spectral_sensitivity.intensities, fill_value='extrapolate')
        lippmann_discrete.spectrums.intensities = f1(new_wave_lengths)
        lippmann_discrete.light_spectrum.intensities = f2(new_wave_lengths)
        lippmann_discrete.spectral_sensitivity.intensities = f3(new_wave_lengths)
        
        return lippmann_discrete
        
class LippmannPixelDiscrete(object):
    
    def __init__(self, intensities, reverse=False, nu_min=43., nu_max=77., light_spectrum=None, spectral_sensitivity=None, n=1.):
        
        self.N_prime = len(intensities)
        self.N = np.floor(self.N_prime*nu_max/(nu_max-nu_min))
            
        self.f_max = 2./(390E-9)
        self.df    = self.f_max/(self.N-1)
        self.dr    = 1./(2.*self.f_max)  
        
        self.f           = np.arange(self.N)*self.df
        self.lambdas     = 2./self.f[-self.N_prime:]
        if reverse:
            self.f       = self.f[::-1]
            self.lambdas = self.lambdas[::-1]
        

        self.z           = np.arange(self.N)*self.dr
        
        self.object_spectrum = Spectrum(self.lambdas, intensities)
                
        if light_spectrum is None:
            self.light_spectrum = Spectrum( self.object_spectrum.wave_lengths, np.ones(self.N_prime) )
        else:
            self.light_spectrum = light_spectrum
    
        if spectral_sensitivity is None:
            self.spectral_sensitivity = Spectrum( self.object_spectrum.wave_lengths, np.ones(self.N_prime) )
        else:
            self.spectral_sensitivity = spectral_sensitivity
            
        self.n  = n
        self.c0 = 299792458
        self.c  = self.c0/n
        self.epsilon_0 = 8.8541878176E-12
            
        self.reflectivity = None
        self.intensity    = None
    
    
    def get_intensity(self):
        
        if not self.intensity is None:
            return self.intensity

        #pad zeros to account for the lowpass band
        x_i = np.hstack( [ np.zeros(self.N-self.N_prime), self.light_spectrum.intensities*self.object_spectrum.intensities ] )
        intensity = np.sum(x_i)/self.N - dct(x_i)      
        
        self.intensity = Spectrogram(self.z, intensity)
        
        return self.intensity
        
    def get_reflectivity(self):
        
        if not self.reflectivity is None:
            return self.reflectivity

        #pad zeros to account for the lowpass band
        x_r = np.hstack( [ np.zeros(self.N-self.N_prime), self.light_spectrum.intensities*self.spectral_sensitivity.intensities*self.object_spectrum.intensities ] )
        g = dct(x_r, type=1)
        R = g[0] - g
                
        self.reflectivity = Spectrogram(self.z, R)
        
        return self.reflectivity
        
    def re_illuminate(self, new_light_spectrum=None):
        
    
        if new_light_spectrum is None:
            new_light_spectrum = self.light_spectrum
            
        if not self.reflectivity is None:
            self.get_reflectivity()
            
        #Normalize    
        G = 1./(2*(self.N-1))*dct(self.reflectivity.intensities, type=1)
        
        return Spectrum(self.lambdas, new_light_spectrum.intensities*G[-self.N_prime:]**2)
    

class LippmannPlateDiscrete(object):
    
    def __init__(self, N_prime, n_x, n_y, light_spectrum=None, spectral_sensitivity=None, n=1.0, E_0=1.0, phi_0=np.pi/2.0):
        
        self.N_prime = N_prime
        self.N = np.floor(N_prime*77./34.)
            
        self.f_max = 2./(390E-9)
        self.df    = self.f_max/(self.N-1)
        self.dr    = 1./(2.*self.f_max)  
        
        self.f           = np.arange(self.N)*self.df
        self.lambdas     = 2./self.f[-self.N_prime:]

        self.z           = np.arange(self.N)*self.dr
        self.r           = np.zeros([self.N, 3])
        self.r[:,2]      = self.z
        
        self.width  = n_x
        self.height = n_y
        self.spectrums = Spectrums(self.lambdas, np.zeros([n_x, n_y, N_prime]))      
        
        self.E_0 = E_0
        self.phi_0 = phi_0
        self.A = E_0*np.exp(1j*phi_0)       #complex envelope
        self.n = n

        #speed of light 
        self.c0 = 299792458
        self.c  = self.c0/n
        self.epsilon_0 = 8.8541878176E-12
        
        if light_spectrum is None:
            self.light_spectrum = Spectrum( self.lambdas, np.ones(N_prime) )
        else:
            self.light_spectrum = light_spectrum
    
        if spectral_sensitivity is None:
            self.spectral_sensitivity = Spectrum( self.lambdas, np.ones(N_prime) )
        else:
            self.spectral_sensitivity = spectral_sensitivity
        
        self.intensities   = None
        self.reflectances  = None
        self.new_spectrums = None
        
        
    def compute_intensity(self):
        
        if self.intensities is None:
            
            self.intensities    = np.zeros([self.width, self.height, self.N])
            self.reflectances   = np.zeros([self.width, self.height, self.N])
            
            for x in xrange(self.width):
                print x
                for y in xrange(self.height):
                    
                    #pad zeros to account for the lowpass band
                    x_i = np.hstack( [ np.zeros(self.N-self.N_prime), self.spectrums.intensities[x, y, :]*self.light_spectrum.intensities ] )
                    x_r = np.hstack( [ np.zeros(self.N-self.N_prime), self.spectrums.intensities[x, y, :]*self.spectral_sensitivity.intensities*self.light_spectrum.intensities ] )

                    g_i = dct(x=x_i, type=1)
                    g_r = dct(x=x_r, type=1)

                    self.intensities[x, y,:]  = g_i[0] - g_i
                    self.reflectances[x, y,:] = g_r[0] - g_r 
                    
    def compute_new_spectrum(self):
                
        if self.new_spectrums is None:
            
            if self.reflectances is None:
                self.compute_intensity()           
                
            self.new_spectrums = Spectrums( self.lambdas, np.zeros([self.width, self.height, self.N_prime]) )
   
                
            for x in xrange(self.width):
                print x
                for y in xrange(self.height):
                    
                    #Normalize    
                    G = 1./(2*(self.N-1))*dct(self.reflectances[x,y,:], type=1)
                    self.new_spectrums.set_spectrum( x, y, self.light_spectrum.intensities*G[-self.N_prime:]**2 )
            
    