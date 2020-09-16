# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 14:01:34 2017

@author: gbaechle
"""


import numpy as np
import scipy as sp
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.optimize import least_squares

import seaborn as sns
import seabornstyle as snsty

import sys
sys.path.append("../")
from multilayer_optics_matrix_theory import *
import color_tools as ct
import finite_depth_analysis as fda

snsty.setStyleMinorProject()
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


plt.close('all')
fig_path = 'Figures/'


c0 = 299792458
n0 = 1.5
c = c0/n0


def generate_gaussian_spectrum(lambdas, mu=550E-9, sigma=30E-9):
    spectrum = sp.stats.norm(loc=mu, scale=sigma).pdf(lambdas)

    return spectrum/np.max(spectrum)

def generate_rect_spectrum(lambdas, start=450E-9, end=560E-9):
    spectrum = np.zeros(len(lambdas))
    spectrum[(lambdas >= start) & (lambdas <= end)] = 1

    return spectrum

def generate_mono_spectrum(lambdas, color=550E-9):
    spectrum = np.zeros(len(lambdas))
    spectrum[np.argmin(np.abs(lambdas - color))] = 1

    return spectrum

def lippmann_transform_complex(lambdas, spectrum, depths, r=-1, t=0):
    
    k = 2 * np.pi / lambdas
    omegas = 2*np.pi*c/lambdas

    return -np.trapz( np.sqrt(spectrum[None, :])*np.exp(-1j*omegas[None, :]*t)*(np.exp(-1j*k[None, :]*depths[:, None]) + r*np.exp(1j*k[None, :]*depths[:, None])), omegas, axis=1)


def lippmann_transform(lambdas, spectrum, depths, r=-1, k0=0):
    two_k = 4 * np.pi / lambdas
    omegas = 2*np.pi*c/lambdas
    theta = np.angle(r)
    rho   = np.abs(r)

    cosine_term = (1 + rho**2 + 2*rho*np.cos(two_k[None, :] * depths[:, None] + theta))
    cosines = 2*rho*np.cos(two_k[None, :] * depths[:, None] + theta)

    intensity = -np.trapz(cosine_term * spectrum[None, :], omegas, axis=1)
    delta_intensity = -np.trapz(cosines * spectrum[None, :], omegas, axis=1)
    
    window = np.exp(-k0/np.max(depths)*depths)
    
    return intensity*window, delta_intensity

def inverse_lippmann(intensity, lambdas, depths, symmetric=False, return_intensity=True):
    two_k = 4 * np.pi / lambdas
    
    if symmetric:
        I = np.r_[intensity[:0:-1], intensity]
        d = np.r_[-depths[:0:-1], depths]
    else:
        I = intensity
        d = depths

    exponentials = np.exp(-1j * two_k[:, None] * d[None, :])
    if return_intensity:
        return np.abs(np.trapz(exponentials * I[None, :], d, axis=1)) ** 2
    else:
        return np.trapz(exponentials * I[None, :], d, axis=1)


def lippmann_transform_reverse(lambdas, intensity, depths, r=-1):
    two_k = 4 * np.pi / lambdas
    theta = np.angle(r)
    
    nu = 2*np.mod(-theta, 2*np.pi)/np.pi
    
    I = intensity - np.mean(intensity)
    
    integrand = (two_k[:,None]*depths[None,:])**nu * \
                (sp.special.hyp1f1(1,1+nu, 1j*two_k[:,None]*depths[None,:]) + \
                 sp.special.hyp1f1(1,1+nu, -1j*two_k[:,None]*depths[None,:]))
                
    return 2/(c*np.pi*sp.special.gamma(nu+1)) * np.trapz(integrand * I[None,:], depths, axis=1)
    

def inverse_lippmann_reverse(depths, spectrum, lambdas, initial_estimate=None):
    
    if initial_estimate is None:
        initial_estimate = np.ones_like(depths)

    lippmann_error = lambda intensity: inverse_lippmann(intensity, lambdas, depths) - spectrum
    return least_squares(fun=lippmann_error, x0=initial_estimate, verbose=2, max_nfev=300, gtol=1E-100, method='trf', bounds=(0,np.inf)).x


def apply_h_reverse(lambdas, spectrum, Z, r=-1, initial_estimate=None):
    
    if initial_estimate is None:
        initial_estimate = np.ones_like(depths)

    lippmann_error = lambda original: np.abs( fda.apply_h(original, Z, lambdas, lambdas, r=r) )**2 - spectrum
    return least_squares(fun=lippmann_error, x0=initial_estimate, verbose=2, max_nfev=300, gtol=1E-100, method='trf', bounds=(0,np.inf)).x




def plot_gaussian_lippmann_and_inverses():  
    
    lambdas, omegas = generate_wavelengths(500) #3000
    depths = generate_depths(max_depth=5E-6)
    
    spectrum = generate_gaussian_spectrum(lambdas=lambdas, mu=550E-9, sigma=30E-9)
    
    lippmann = plot_spectrums_and_methods(lambdas, depths, spectrum, n0, name='gaussian')
    
    return lambdas, depths, spectrum, lippmann
    
    
def plot_mono_lippmann_and_inverses():   
    
    lambdas, omegas = generate_wavelengths(500)
    depths = generate_depths(max_depth=5E-6)
    
    spectrum = generate_mono_spectrum(lambdas=lambdas, color=600E-9)
    
    lippmann = plot_spectrums_and_methods(lambdas, depths, spectrum, n0, name='mono')
    
    return lambdas, depths, spectrum, lippmann
    
    
def plot_rect_lippmann_and_inverses():  
    
    lambdas, omegas = generate_wavelengths(500)
    depths = generate_depths(max_depth=5E-6)
    
    spectrum = generate_rect_spectrum(lambdas=lambdas, start=480E-9, end=580E-9)
    
    lippmann = plot_spectrums_and_methods(lambdas, depths, spectrum, n0, name='rect')
     
    return lambdas, depths, spectrum, lippmann
    
      
def plot_spectrums_and_methods(lambdas, depths, spectrum, n0, name):
    
    f, (ax1, ax2) = plt.subplots(1, 2)
    
    show_spectrum(lambdas, spectrum, ax=ax1)
     
    lippmann, delta_lippmann = lippmann_transform(lambdas/n0, spectrum, depths)
    show_lippmann_transform(depths, lippmann, ax=ax2)
    plt.savefig(fig_path + name + '.pdf') 
    plt.show()
    
    spectrum_inverse_lippmann = inverse_lippmann(lippmann, lambdas/n0, depths)
    plt.figure()    
    show_spectrum(lambdas, spectrum_inverse_lippmann, ax=plt.gca())
    plt.savefig(fig_path + name + '_inverse_lippmann.pdf') 
    plt.show()
    
    ns = generate_lippmann_refraction_indices(delta_lippmann, n0=n0, mu_n=0.01)
    spectrum_inverse_nareid, _ = propagation_arbitrary_layers_spectrum(ns, d=10E-9, lambdas=lambdas, plot=False)
    
    plt.figure()    
    show_spectrum(lambdas, spectrum_inverse_nareid, ax=plt.gca())
    plt.savefig(fig_path + name + '_inverse_nareid.pdf') 
    plt.show() 
    
    return lippmann
      
def show_lippmann_transform(depths, lippmann, ax=None, black_theme=False, complex_valued=False, nolabel=False, short_display=False, label=''):
    
    z = depths*1E6
    
    if ax is None:
        plt.figure()
        ax=plt.gca()
  
    vmax = 1.1*np.max(np.abs(lippmann))
    ax.plot(z, np.real(lippmann), linewidth=1.0, zorder=-1, alpha=1.)
    if complex_valued:
        ax.plot(z, np.imag(lippmann), linewidth=1.0, zorder=-2, alpha=1., c='0.7')
        ax.set_ylim(-vmax, vmax)
        ax.set_yticks([0])
    else:
        ax.set_ylim(0, vmax)
        ax.set_yticks([])
    ax.set_xlim([np.min(z), np.max(z)])
    
    if black_theme:
        ax.set_xticks([])
        
    if not nolabel and not short_display:
        ax.set_xlabel('Depth ($\mu m$)')
       
    if not nolabel and short_display:
        ax.set_xticks([0, depths[-1]*1E6/2, depths[-1]*1E6])
        ax.set_xticklabels([0, 'Depth ($\mu m$)', int(np.round(depths[-1]*1E6))])  
        ax.set_xlabel(label)
    

def show_spectrum(lambdas, spectrum, ax=None, visible=False, true_spectrum=True, vmax=None, show_background=False, lw=1, nolabel=False, short_display=False, label=''):
    
    if visible:
        spectrum = spectrum[(lambdas <= 700E-9) & (lambdas >= 400E-9)]
        lambdas = lambdas[(lambdas <= 700E-9) & (lambdas >= 400E-9)]
    
    lam = lambdas*1E9
    
    if ax is None:
        plt.figure()
        ax=plt.gca()
    
    L = len(lam)
    
    if true_spectrum:
        cs = [wavelength_to_rgb(wavelength) for wavelength in lam]
#        cs = lambdas_to_rgb(lambdas)
    else:
        colors = plt.cm.Spectral_r(np.linspace(0, 1, L))
        cs = [colors[i] for i in range(L)]
    
#    ax.scatter(lam, spectrum, color=cs, s=10)
#    ax.plot(lam, spectrum, '--k', linewidth=0.5, zorder=-1, alpha=0.5, dashes=(2,2))
    fda.plot_gradient_line(lam, spectrum, ax=ax, zorder=-1, lw=lw, cs=cs)
    ax.set_xlim([np.min(lam), np.max(lam)])
    if vmax is None:
        vmax = 1.1*np.max(spectrum)
        
    ax.set_ylim(0, vmax)
    ax.set_yticks([])
    
    if not nolabel and not short_display:
        ax.set_xticks([400, 500, 600, 700])
        ax.set_xlabel('Wavelength ($nm$)')
        
    if not nolabel and short_display:
        ax.set_xticks([400, 550, 700])
        ax.set_xticklabels([400, '$\lambda$ ($nm$)', 700])
        ax.set_xlabel(label)
        
    
    if show_background:
        col = ct.from_xyz_to_rgb( ct.from_spectrum_to_xyz(lambdas, spectrum).reshape((1,1,-1)) ).flatten()
        ax.add_patch(patches.Rectangle((lam[0], 0.95*vmax), lam[-1]-lam[0], 1.05*vmax, facecolor=col, edgecolor='none', zorder=-10))
#        ax.set_facecolor(col)
    
def lambdas_to_rgb(lambdas):
    
    colors_XYZ = np.zeros((len(lambdas), 3))
    for i, lambd in enumerate( lambdas ):
    
        spectrum = generate_mono_spectrum(lambdas=lambdas, color=lambd)
        colors_XYZ[i, :] = -ct.from_spectrum_to_xyz(lambdas, spectrum, normalize=False)
        

    #normalize colors
    colors_xyz = colors_XYZ/np.max(colors_XYZ[:,1])
    colors_rgb = ct.from_xyz_to_rgb(colors_xyz.reshape((-1,1,3)), normalize=False).reshape((-1,3))
    colors_rgb /= np.max(colors_rgb)
    
    return colors_rgb
    
def wavelength_to_rgb(wavelength, gamma=0.8):
    '''This converts a given wavelength of light to an 
    approximate RGB color value. The wavelength must be given
    in nanometers in the range from 380 nm through 750 nm
    (789 THz through 400 THz).

    Based on code by Dan Bruton
    http://www.physics.sfasu.edu/astro/color/spectra.html
    '''

    wavelength = float(wavelength)
    if wavelength >= 380 and wavelength <= 440:
        attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
        R = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma
        G = 0.0
        B = (1.0 * attenuation) ** gamma
    elif wavelength >= 440 and wavelength <= 490:
        R = 0.0
        G = ((wavelength - 440) / (490 - 440)) ** gamma
        B = 1.0
    elif wavelength >= 490 and wavelength <= 510:
        R = 0.0
        G = 1.0
        B = (-(wavelength - 510) / (510 - 490)) ** gamma
    elif wavelength >= 510 and wavelength <= 580:
        R = ((wavelength - 510) / (580 - 510)) ** gamma
        G = 1.0
        B = 0.0
    elif wavelength >= 580 and wavelength <= 645:
        R = 1.0
        G = (-(wavelength - 645) / (645 - 580)) ** gamma
        B = 0.0
    elif wavelength >= 645 and wavelength <= 750:
        attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
        R = (1.0 * attenuation) ** gamma
        G = 0.0
        B = 0.0
    else:
        R = 0.0
        G = 0.0
        B = 0.0
    R *= 255
    G *= 255
    B *= 255
    return np.array([int(R), int(G), int(B), 255])/255.


def generate_wavelengths(N=300, omega_low=None, c=c):
    
    lambda_low = 390E-9
    lambda_high = 700E-9
    if omega_low is None:
        omega_low = 2 * np.pi * c / lambda_high
    omega_high = 2 * np.pi * c / lambda_low
    omegas = np.linspace(omega_high, omega_low, N)
    lambdas = 2 * np.pi * c / omegas

    return lambdas, omegas


def generate_wavelengths_sinc(Z, omega_low=None, symmetric=False, c=c):
    
    lambda_low = 390E-9
    lambda_high = 700E-9

    if omega_low is None:
        omega_low = 2 * np.pi * c / lambda_high
    omega_high = 2 * np.pi * c / lambda_low
    
    if symmetric:
        omegas = np.arange(0, omega_high, np.pi*c/(2*Z))[::-1]
    else:
        omegas = np.arange(0, omega_high, np.pi*c/(Z))[::-1]
    omegas = omegas[omegas >= omega_low]
    
    lambdas = 2 * np.pi * c / omegas

    return lambdas, omegas   

def generate_wavelengths_sinc_new(Z, omegas, c=c): 
    
    mi = np.min(omegas)
    ma = np.max(omegas)
    
    Z *= 2
    
    if mi < 0:
        omegas_sinc_p = np.arange(0, ma, np.pi*c/Z)
        omegas_sinc_n = np.arange(0, mi, -np.pi*c/Z)[::-1]
        omegas_sinc = np.r_[omegas_sinc_n[:-1], omegas_sinc_p]
    else:
        omegas_sinc = np.arange(0, ma, np.pi*c/Z)
        omegas_sinc = omegas_sinc[omegas_sinc >= mi]
        
    omegas_sinc = omegas_sinc[::-1]
    lambdas_sinc = 2 * np.pi * c / omegas_sinc
    
    return lambdas_sinc, omegas_sinc
        
    
def generate_depths(delta_z=10E-9, max_depth=10E-6):
    
    return np.arange(0, max_depth-delta_z, delta_z)


if __name__ == '__main__':
    
    plt.close('all') 
    
    Z = 5E-6
    N = 500
    
    r = 0.5*np.exp(-1j*1.256)
    r = -1
    
    lambdas, omegas = generate_wavelengths(N) #3000
    depths = np.linspace(0,Z*(1-1/N),N)
#    depths = generate_depths(max_depth=5E-6)
    omegas = 2 * np.pi * c / lambdas 
    
    spectrum = generate_gaussian_spectrum(lambdas=lambdas, mu=550E-9, sigma=30E-9)
#    spectrum = generate_gaussian_spectrum(lambdas, mu=650E-9, sigma=30E-9) + 1.4*generate_gaussian_spectrum(lambdas, mu=450E-9, sigma=20E-9)
#    spectrum = generate_mono_spectrum(lambdas=lambdas, color=530E-9)
    spectrum /= np.max(spectrum)
    
    lippmann, delta_lippmann = lippmann_transform(lambdas, spectrum, depths, r=r)   
#    lippmann /= np.max(lippmann)
    
    spectrum_reconstructed = lippmann_transform_reverse(lambdas, lippmann, depths, r=r)
        
#    f, (ax1, ax2, ax3) = plt.subplots(1, 3)
#    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(1.2*3.45/0.6, 0.7*3.45/0.6))
    f, (ax1, ax2) = plt.subplots(1, 2)
    show_spectrum(lambdas, spectrum, ax=ax1, short_display=True)
    show_lippmann_transform(depths, lippmann, ax=ax2, short_display=True)
    ax1.set_xlabel('(a) Spectrum')
    ax2.set_xlabel('(b) Interference patterns')
    plt.savefig(fig_path + 'gaussian.pdf')
#    show_spectrum(lambdas, spectrum_reconstructed, ax=ax3)
    
        
    spectrum_filtered =  fda.apply_h(spectrum, Z, lambdas, lambdas, r=r)
    spectrum_reconstructed = inverse_lippmann(lippmann, lambdas, depths, symmetric=False)
    lippmann_retransform, redelta_lippmann = lippmann_transform(lambdas, spectrum_reconstructed, depths, r=r)
#    lippmann_retransform = inverse_inverse_lippmann(spectrum_reconstructed**2, lambdas/n0, depths)  
    
#    C = np.trapz(np.sqrt(spectrum_reconstructed), omegas)
#    C = -np.trapz(spectrum, omegas)/2
#    box_spectrum = inverse_lippmann(C*np.ones_like(depths), lambdas, depths)
#    
#    spectrum_cleaned = (np.sqrt(spectrum_reconstructed)-np.sqrt(box_spectrum))**2
    
#    spectrum_reverse = lippmann_transform_reverse(lambdas, lippmann, depths, r=r)
#    lippmann_reverse = inverse_lippmann_reverse(depths, spectrum_reconstructed, lambdas, initial_estimate=np.maximum( lippmann_retransform/np.max(lippmann_retransform)*np.max(lippmann), 0 ))
#    filtered_reverse = apply_h_reverse(lambdas, spectrum_filtered, Z, r=-1, initial_estimate=spectrum_filtered/np.max(spectrum_filtered)*np.max(spectrum))
    
    f, axes = plt.subplots(2,2, figsize=(1.2*3.45/0.6, 0.7*3.45/0.6), sharex='col')
    show_spectrum(lambdas, spectrum, ax=axes[0,0], true_spectrum=False)
    show_spectrum(lambdas, np.abs(spectrum_filtered)**2, ax=axes[0,1], true_spectrum=False)
    plt.savefig('lippmann_pairs.pdf')
    
#    spectrum_reverse = np.load('spectrum_reverse.npy')
    lippmann_reverse = np.load('lippmann_reverse.npy')
#    filtered_reverse = np.load('filtered_reverse.npy')
#    
#    
#    
#    f, axes = plt.subplots(2, 2, figsize=(3.45, 3.45), sharex='col')
#    show_spectrum(lambdas, spectrum, ax=axes[0,0]) 
#    axes[0,0].set_xlabel('')
#    show_lippmann_transform(depths, lippmann, ax=axes[0,1]) 
#    axes[0,1].set_xlabel('')
#    show_spectrum(lambdas, spectrum_reverse, ax=axes[1,0]) 
#    show_lippmann_transform(depths, lippmann_transform(lambdas, spectrum_reverse, depths, r=r)[0], ax=axes[1,1]) 
#    axes[1,1].set_xticks(np.arange(5))
#    plt.savefig('same_lippmann.pdf')
#    
    f, axes = plt.subplots(2, 2, figsize=(3.45/0.5*0.6, 3.45/0.5*0.6), sharex='col')
    show_lippmann_transform(depths, lippmann, ax=axes[0,0])
    axes[0,0].set_xlabel('')
    show_spectrum(lambdas, spectrum_reconstructed, ax=axes[0,1]) 
    axes[0,1].set_xlabel('')
    show_lippmann_transform(depths, lippmann_reverse, ax=axes[1,0]) 
    show_spectrum(lambdas, inverse_lippmann(lippmann_reverse, lambdas, depths), ax=axes[1,1]) 
    axes[1,0].set_xticks(np.arange(5))
    plt.savefig('same_spectrum.pdf')
#    
#    f, axes = plt.subplots(2, 2, figsize=(3.45, 3.45), sharex='col')
#    show_spectrum(lambdas, spectrum, ax=axes[0,0])
#    show_spectrum(lambdas, spectrum_reconstructed, ax=axes[0,1]) 
#    show_spectrum(lambdas, filtered_reverse, ax=axes[1,0]) 
#    show_spectrum(lambdas, np.abs( fda.apply_h(spectrum, Z, lambdas, lambdas, r=-1) )**2, ax=axes[1,1]) 
#    plt.savefig('same_filtered.pdf')
#
#
#    f, (ax1, ax2) = plt.subplots(1, 2)
#    show_spectrum(lambdas, spectrum, ax=ax1, label='(a) Spectrum')
#    show_lippmann_transform(depths, lippmann, ax=ax2, short_display=True, label='(b) Interference patterns')
#
#    plt.savefig(fig_path + 'gaussian.pdf') 

    
#    plt.figure()
#    show_lippmann_transform(depths, lippmann, ax=plt.gca()) 
#    plt.savefig('lippmann.pdf')
#    plt.title('Lippmann transform')
    
#    plt.figure()
#    show_lippmann_transform(depths, lippmann_retransform, ax=plt.gca()) 
#    plt.title('Lippmann retransform')    
    
#    plt.figure()
#    show_spectrum(lambdas, spectrum, ax=plt.gca()) 
#    plt.title('Original spectrum') 
    
#    plt.figure()
#    show_spectrum(lambdas, spectrum_reconstructed, ax=plt.gca()) 
#    plt.title('Spectrum reconstructed (inverse Lippmann)') 
    


    
#    plt.figure()
#    show_spectrum(lambdas, box_spectrum, ax=plt.gca()) 
#    plt.title('Spectrum box') 
    
#    plt.figure()
#    show_spectrum(lambdas, spectrum_cleaned, ax=plt.gca()) 
#    plt.title('Spectrum cleaned') 

    
    
#    plt.figure()
#    show_spectrum(lambdas, np.real(spectrum_amp), ax=plt.gca()) 
#    plt.title('Spectrum reconstructed (amplitude)') 
    
    
    
#    plt.figure()
#    show_spectrum(lambdas, spectrum_re_reconstructed, ax=plt.gca()) 
#    plt.title('Spectrum re-reconstructed (inverse Lippmann)') 
    
#    spectrum_cosine = inverse_lippmann_cosine(lippmann_retransform, lambdas/n0, depths)
#    plt.figure()
#    show_spectrum(lambdas, spectrum_cosine, ax=plt.gca()) 
#    plt.title('Spectrum inverse cosine') 
    
    
#    plot_mono_lippmann_and_inverses()
#    plot_rect_lippmann_and_inverses()
    
    
