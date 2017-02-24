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
    
    def __init__(self, object_spectrum, n=1.0, light_spectrum=None, spectral_sensitivity=None):
        
        wave_lengths = object_spectrum.wave_lengths
        l = len(wave_lengths)  
        
        self._U = None
        self._I = None
        self._R = None
        
        #speed of light 
        self.c0 = 299792458
        self.c  = self.c0/n
        self.epsilon_0 = 8.8541878176E-12
        
        #self.direction = direction
        self.ks = 2.0*np.pi/wave_lengths      #wavevector
        self.omegas = self.ks*self.c 
        
        if light_spectrum is None:
            self.light_spectrum = Spectrum( wave_lengths, np.ones(l) )
        else:
            self.light_spectrum = light_spectrum
    
        if spectral_sensitivity is None:
            self.spectral_sensitivity = Spectrum( wave_lengths, np.ones(l) )
        else:
            self.spectral_sensitivity = spectral_sensitivity   
        
    def compute_light_field(self):
        """Compute the incoming light field."""
        raise NotImplementedError( "This function should be implemented" )        
        
    def compute_intensity(self):
        """Compute the intensity of the interference field as well as the reflectivity of the Lippmann plate"""
        raise NotImplementedError( "This function should be implemented" )
        
    def compute_new_spectrum(self):
        """Compute the intensity of the interference field as well as the reflectivity of the Lippmann plate"""
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
    def U(self):
        if self._U is None:
            self.compute_light_field()
        return self._U
        
    @U.setter
    def U(self, value):
        self._U = value
        
    #I = property(fget=_get_intensity, fset=_set_intensity, doc="""Gets or sets the intensity.""")
    #R = property(fget=_get_reflectivity, fset=_set_reflectivity, doc="""Gets or sets the reflectivity.""")
    #U = property(fget=_get_light_field, fset=_set_light_field, doc="""Gets or sets the light field.""")


#class LippmannPlate(object):
#    """Class defining a Lippmann object for a 2D image (i.e. with spatial variation)"""


        
        
       
class LippmannPlate(object):
    """Class defining a Lippmann object for a 2D image (i.e. with spatial variation)"""
    
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
            
            for x in range(self.width):
                perc = np.double(x)/np.double(self.width-1)*100
                sys.stdout.write("\rComputing intensity: %.2f %%" %perc)
                sys.stdout.flush()                
                
                for y in range(self.height):
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
                            
            for x in range(self.width):
                perc = np.double(x)/np.double(self.width-1)*100
                sys.stdout.write("\rComputing new spectrum: %.2f %%" %perc)
                sys.stdout.flush() 
                
                for y in xrange(self.height):
                    
                    integrand = self.reflectances[x, y, :, np.newaxis] * cosines.T
                    
#                    self.new_spectrums.set_spectrum( x, y, 0.5*self.c*self.epsilon_0*self.light_spectrum.intensities * np.trapz(y=integrand, x=self.r[:,2], axis=0)**2 )
                    self.new_spectrums.set_spectrum( x, y, 1E20*self.light_spectrum.intensities * np.trapz(y=integrand, x=self.r[:,2], axis=0)**2 )
            
            
    def to_uniform_freq(self, N_prime):
    
        lambda_max = np.max(self.spectrums.wave_lengths)
        lippmann_discrete = LippmannPlateDiscrete(N_prime, self.width, self.height, lambda_max=lambda_max,  light_spectrum=self.light_spectrum, \
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
    
    def __init__(self, N_prime, n_x, n_y, lambda_max=700E-9, light_spectrum = None, spectral_sensitivity=None, n=1.0, E_0=1.0, phi_0=np.pi/2.0):
        
        #speed of light 
        self.c0 = 299792458
        self.c  = self.c0/n
        self.epsilon_0 = 8.8541878176E-12
        
        #reds (or higher)
        v_min = self.c/lambda_max
        #blues
        lambda_min = 390E-9
        v_max = self.c/lambda_min
        
        self.N_prime = N_prime
        self.N = np.int( np.floor(N_prime*v_max/(v_max-v_min)) )
            
        self.f_max = 2./(390E-9)
        self.df    = self.f_max/(self.N-1)
        self.dr    = 1./(2.*self.f_max)          
        
#        self.N_prime = N_prime
#        self.N = np.int( np.floor(N_prime*77./34.) )
#            
#        self.f_max = 2./(390E-9)
#        self.df    = self.f_max/(self.N-1)
#        self.dr    = 1./(2.*self.f_max)  
        
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
            
#            sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)
            
            for x in range(self.width):
                perc = np.double(x)/np.double(self.width-1)*100
                sys.stdout.write("\rComputing intensity: %.2f %%" %perc)
                sys.stdout.flush() 
                
                for y in range(self.height):
                    
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
   
   
            for x in range(self.width):
                perc = np.double(x)/np.double(self.width-1)*100
                sys.stdout.write("\rComputing new spectrum: %.2f %%" %perc)
                sys.stdout.flush()
                
                for y in range(self.height):
                    
                    #Normalize    
                    G = 1./(2*(self.N-1))*dct(self.reflectances[x,y,:], type=1)
                    self.new_spectrums.set_spectrum( x, y, self.light_spectrum.intensities*G[-self.N_prime:]**2 )
            
    