# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 21:40:59 2017

@author: gbaechle
"""

import numpy as np
import scipy as sp
import scipy.interpolate
import scipy.fftpack
import scipy.signal
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches

from lippmann import *

import seabornstyle as snsty

black_theme=False

#snsty.setStyleMinorProject(black=black_theme)
snsty.setStylePNAS()

import multilayer_optics_matrix_theory as matrix_theory

plt.rc('text', usetex=True)
#plt.rc('font', family='serif')

c0 = 299792458
n0 = 1.45
c = c0/n0

def get_axis(ax=None):
    
    if ax is None:
        plt.figure()
        ax=plt.gca()
        
    return ax

def sinc_interp(x, s, u):
   
#    Interpolates x, sampled at "s" instants
#    Output y is sampled at "u" instants ("u" for "upsampled") 
   
    T = s[1] - s[0] # Find the period 
    
    sincM = np.tile(u, (len(s), 1)) - np.tile(s[:, None], (1, len(u)))
    y = x @ np.sinc(sincM/T)
    return y

    
def s_z(Z, omega):
    
    #note: numpy sinc is sin(pi*x)/(pi*x)
    return Z*np.sinc(2*Z*omega/(np.pi*c))
    
def c_z(Z, omega):
    
    x = np.asanyarray(2*Z*omega/c)
    y = np.where(x == 0, 1.0e-20, x) 
    
    return Z*(1-np.cos(y))/y    
    
    
def s_z_tilde(Z, omega):
    
    x = np.asanyarray(2*Z*omega/c)
    y = np.where(x == 0, 1.0e-30, x) 
    
    return Z/(1j*y)*(1-np.exp(-1j*y))
#    return s_z(Z, omega) + 1j*c_z(Z, omega)
    
def c_inf(omega):
    x = np.asanyarray(2*omega/c)
    return np.where(np.abs(x) < 1E-10, 0, 1/x)     
    
def c_high(omega):
    Z = 100E-6
    x = np.asanyarray(2*Z*omega/c)
    y = np.where(x == 0, 1.0e-20, x) 
    
    return Z*(1-np.cos(y))/y   
    
def e_inf(omega):
    x = np.asanyarray(2*omega/c, dtype=np.complex)
    return np.where(x == 0, 1, 1j/x) 

    
def h(lambdas, lambda_prime, Z, r=-1, nareid=False):
    
    omega = 2*np.pi*c/lambdas 
    omega_prime = 2*np.pi*c/lambda_prime
    
    x = 2*omega_prime/c
    
    if nareid:
        return r/2*s_z_tilde(Z, omega_prime-omega) + np.conj(r)/2*s_z_tilde(Z, omega+omega_prime)
    else:
#        return (1+np.abs(r)**2)/2*s_z_tilde(Z, omega_prime) 
#        return r/2*s_z_tilde(Z, omega_prime-omega) + np.conj(r)/2*s_z_tilde(Z, omega+omega_prime)
        return r/2*s_z_tilde(Z, omega_prime-omega) + (1+np.abs(r)**2)/2*s_z_tilde(Z, omega_prime) + np.conj(r)/2*s_z_tilde(Z, omega+omega_prime)

      
def h_sym(lambdas, lambda_prime, Z, r=-1):
    
    omega = 2*np.pi*c/lambdas 
    omega_prime = 2*np.pi*c/lambda_prime    
     
    return r/2*s_z(Z, omega-omega_prime) + (1+np.abs(r)**2)/2*s_z(Z, omega_prime) + np.conj(r)/2*s_z(Z, omega+omega_prime)                     
    
def plot_h(lambdas, filt, name='h', lambda_prime=500E-9, Z=5E-6, complex_valued=True, ax=None):
    
    ax = get_axis(ax)
    
    show_spectrum_complex(lambdas, filt, ax=ax, complex_valued=complex_valued)
#    plt.savefig(fig_path + name + '.pdf') 
    
    return filt
    
def show_spectrum_frequencies(omegas, spectrum, ax=None, zorder=1, x_label=True, vmax=None, vmin=0, col=None):
        
    ax = get_axis(ax)
          
    if vmax is None:
        vmax = 1.1*np.max(spectrum[int(len(spectrum)*0.75):])
    ax.set_ylim(vmin, vmax)      
    if c is None:    
        ax.plot(omegas, spectrum, lw=1)
    else:
        ax.plot(omegas, spectrum, lw=1, c=col)
    
    ax.set_xlim([np.min(omegas), np.max(omegas)])
    ax.set_yticks([])
  
    if black_theme:
        ax.set_xticks([])
    elif x_label:
#        ax.set_xticks([400, 500, 600, 700])
        ax.set_xlabel('Frequency $\omega$')
        
    _, omegas_visible = generate_wavelengths(1000, c=c)
    
    rect = patches.Rectangle((omegas_visible[-1], vmin), omegas_visible[0]-omegas_visible[-1], 2*vmax, facecolor='0.9', edgecolor='none', zorder=-10)

    ax.add_patch(rect)
    lw = 20 if (vmin==0) else 10*np.sqrt(2)
    plot_gradient_line(omegas_visible, np.ones_like(omegas_visible)*vmax*1.05, ax=ax, lw=lw, plot_points=False)
            
    
    
def show_spectrum_complex(lambdas, spectrum, ax=None, complex_valued=True, true_spectrum=True, intensity=False, zorder=1, lw=2, c=None):
    
    lam = lambdas*1E9
    
    ax = get_axis(ax)
    
    L = len(lam)
    
    if true_spectrum:
        cs = [wavelength_to_rgb(wavelength) for wavelength in lam]
    else:
        colors = plt.cm.Spectral_r(np.linspace(0, 1, L))
        cs = [colors[i] for i in range(L)]
        
    if complex_valued == True:
        if black_theme:
#            ax.scatter(lam, np.imag(spectrum), color='white', s=5, zorder=zorder)
            ax.plot(lam, np.imag(spectrum), color='white', lw=lw, zorder=zorder-0.1)
        else:
#            ax.scatter(lam, np.imag(spectrum), color='0.5', s=5, zorder=zorder)
            ax.plot(lam, np.imag(spectrum), color='0.5', lw=lw, zorder=zorder-0.1)
     
    if intensity:
        ax.set_ylim(0, 1.1*np.max(np.abs(spectrum)))
    else:
        ax.set_ylim(-1.1*np.max(np.abs(spectrum)), 1.1*np.max(np.abs(spectrum)))
        
    plot_gradient_line(lam, np.real(spectrum), ax=ax, zorder=zorder, lw=lw, cs=cs)
#    ax.scatter(lam, np.real(spectrum), color=cs, s=3, zorder=zorder)
    ax.set_xlim([np.min(lam), np.max(lam)])
    ax.set_yticks([])
    if black_theme:
        ax.set_xticks([])
    else:
        ax.set_xticks([400, 500, 600, 700])
        ax.set_xlabel('Wavelength (nm)')
        
        
def plot_gradient_line(x, y, ax=None, zorder=1, lw=2, cs=None, plot_points=True):
    
    L = len(x)    
    
    ax = get_axis(ax)
        
    if cs is None:
#        cmap = plt.get_cmap('Spectral_r')
#        colors = plt.cm.Spectral_r(np.linspace(0, 1, L))
#        cs = [colors[i] for i in range(L)]
        lambda_low = 390E-9
        lambda_high = 700E-9
        omega_low = 2 * np.pi * c / lambda_high
        omega_high = 2 * np.pi * c / lambda_low
        omegas = np.linspace(omega_high, omega_low, L)
        lambdas = 2 * np.pi * c / omegas
        cs = [wavelength_to_rgb(wavelength*1E9) for wavelength in lambdas]
    
    cmap = LinearSegmentedColormap.from_list('test', cs)

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    lc = LineCollection(segments, cmap=cmap, norm=plt.Normalize(0, 1), zorder=zorder)
                                                
    lc.set_array(np.linspace(0, 1, L))
    lc.set_linewidth(lw)
    
    ax.add_collection(lc)
    if plot_points:
        ax.scatter(x, y, color=cs, s=2*(lw/2)**2, zorder=zorder)
    
    
    
def show_sampling_lines(lambdas, spectrum, ax=None, zorder=-1, color='0.5'):
    
    ax = get_axis(ax)
    
    (markerlines, stemlines, baselines) = ax.stem(lambdas*1E9, spectrum, 'k-', markerfmt='k.', zorder=zorder)
    plt.setp(markerlines, color = 'black', markersize=2)
    plt.setp(stemlines, 'markersize', 0.5, color=color)
    plt.setp(baselines, 'linewidth', 0)
    
    
def fill_area_under(lambdas, spectrum, ax=None, zorder=0, color='0.5'):
    
    ax = get_axis(ax)
        
    ax.fill_between(lambdas*1E9, 0, spectrum, facecolor=color, zorder=zorder, interpolate=True)
    
    
def apply_h(spectrum, Z, lambdas, lambdas_prime, symmetric=False, nareid=False, infinite=False, r=-1):
    
    new_spectrum = np.zeros(len(lambdas_prime), dtype=np.complex)
    
    omegas = 2*np.pi*c/lambdas 
    omegas_prime = 2*np.pi*c/lambdas_prime 
        
    if infinite:
        spectrum_prime = np.interp(omegas[::-1], omegas[::-1], spectrum[::-1])[::-1] + np.interp(-omegas_prime[::-1], omegas[::-1], spectrum[::-1])[::-1]

        if symmetric:
            return spectrum_prime

    for i, lambda_prime in enumerate(lambdas_prime):
        
        omega_prime = 2*np.pi*c/lambda_prime      
        
        if infinite:
            filt = r/2*c_high(omega_prime-omegas) + np.conj(r)/2*c_high(omegas+omega_prime)
            if omega_prime < 0:
                new_spectrum[i] = np.conj(r)*c*np.pi/4*spectrum_prime[i]+1j*np.trapz(spectrum*filt, omegas)
            else:
                new_spectrum[i] = r*c*np.pi/4*spectrum_prime[i]+1j*np.trapz(spectrum*filt, omegas)
                
        elif symmetric:
            new_spectrum[i] = -np.trapz(spectrum*h_sym(lambdas, lambda_prime, Z, r=r), omegas)
        
        elif nareid:
            new_spectrum[i] = -np.trapz(spectrum*h(lambdas, lambda_prime, Z, nareid=True, r=r), omegas)
#            new_spectrum[i] = -omega_prime/c*np.trapz(spectrum*h(lambdas, lambda_prime, Z, nareid=False, r=r), omegas)
            
        else:
            new_spectrum[i] = -np.trapz(spectrum*h(lambdas, lambda_prime, Z, r=r), omegas)
    
    if infinite:
        Pw = np.trapz(spectrum, omegas)
        new_spectrum += 1j*(1+np.abs(r)**2)/2*c_inf(omegas_prime)*Pw
    
    return new_spectrum
#    return np.abs(new_spectrum)**2
   
    
def compare_with_lippmann(Z=10E-6, delta_z=10E-9, N=500):
        
    lambdas_visible, omegas_visible = generate_wavelengths(N=N, c=c)
    lambdas, omegas = generate_wavelengths_sinc(Z=Z, c=c)
    
    spectrum = generate_gaussian_spectrum(lambdas_visible, mu=550E-9, sigma=30E-9) #+ generate_gaussian_spectrum(lambdas_visible, mu=650E-9, sigma=20E-9)
#    spectrum = generate_mono_spectrum(lambdas_visible)
#    spectrum = generate_rect_spectrum(lambdas_visible)
    
    depths = generate_depths(delta_z=delta_z, max_depth=Z)
    lippmann, delta_intensity = lippmann_transform(lambdas_visible, spectrum, depths)
    
    print(lambdas_visible.shape, lambdas.shape)
    
    spectrum_replay = np.abs(apply_h(spectrum, Z, lambdas_visible, lambdas))**2
    spectrum_replay_visible = np.abs(apply_h(spectrum, Z, lambdas_visible, lambdas_visible))**2
    
    spectrum_replay_nareid = np.abs(apply_h(spectrum, Z, lambdas_visible, lambdas_visible, nareid=True))**2

    spectrum_interpolated = sp.interpolate.interp1d(omegas, spectrum_replay, kind='quadratic', bounds_error=False, fill_value='extrapolate')(omegas_visible)
    #sinc interpolation    
    spectrum_interpolated = sinc_interp(spectrum_replay, omegas, omegas_visible)    
    
    ns = generate_lippmann_refraction_indices(delta_intensity, n0=n0, mu_n=0.01)
    spectrum_refraction_matrices, t = propagation_arbitrary_layers_spectrum(ns, d=delta_z, lambdas=lambdas_visible, symmetric=False)

    f, (ax1, ax2) = plt.subplots(1, 2)
    
    show_spectrum_complex(lambdas_visible, spectrum, ax=ax1, complex_valued=False, intensity=True)
#    plt.title('original spectrum')
    
    show_lippmann_transform(depths, lippmann, ax=ax2, black_theme=black_theme)
    plt.savefig(fig_path + 'gaussian.pdf') 
#    plt.title('Lippmann transform')
    
    show_spectrum_complex(lambdas, spectrum_replay, complex_valued=False, intensity=True)
    plt.savefig(fig_path + 'spectrum_replay.pdf') 
    plt.title('spectrum replayed sampled')
    
    f, (ax1, ax2) = plt.subplots(1, 2)
    
    show_spectrum_complex(lambdas_visible, spectrum_replay_visible, ax=ax1, complex_valued=False, intensity=True, zorder=2.5)
    show_sampling_lines(lambdas, spectrum_replay, ax=ax1, zorder=-10)
    
    show_spectrum_complex(lambdas_visible, spectrum_interpolated, ax=ax2, complex_valued=False, intensity=True)
    
    ax1.set_xticks([400, 550, 700])
    ax1.set_xticklabels([400, 'Wavelength~(nm)', 700])
    ax1.set_xlabel('(a) Sampling')
    ax2.set_xticks([400, 550, 700])
    ax2.set_xticklabels([400, 'Wavelength~(nm)', 700])  
    ax2.set_xlabel('(b) Interpolation')
    plt.savefig(fig_path + 'spectrum_replay_interpolated.pdf') 
#    plt.title('spectrum replayed interpolated') 
    
    show_spectrum_complex(lambdas_visible, spectrum_replay_visible, complex_valued=False, intensity=True, zorder=4)
    show_sampling_lines(lambdas, spectrum_replay, ax=plt.gca())    
    plt.savefig(fig_path + 'spectrum_sampled.pdf')  
    
    f, (ax1, ax2) = plt.subplots(1, 2)
    
    show_spectrum_complex(lambdas_visible, spectrum_replay_visible, ax=ax1, complex_valued=False, intensity=True)    
    show_spectrum_complex(lambdas_visible, spectrum_replay_nareid, ax=ax2, complex_valued=False, intensity=True)
    plt.savefig(fig_path + 'all_inverses.pdf')  
    
    show_spectrum_complex(lambdas_visible, spectrum_interpolated, complex_valued=False, intensity=True)
    

def plt_spectrum(omega, spec, ax=None, x_label=True, complex_valued=False, vmax=None, vmin=0, freq=True, col=None):
    if freq:
        show_spectrum_frequencies(omega[::-1], np.real(spec[::-1]), ax=ax, x_label=x_label, vmax=vmax, vmin=vmin, col=col)
        if complex_valued:
            ax.plot(omega[::-1], np.imag(spec[::-1]), ':', c=col, zorder=-1)
#                ax.set_ylim(-np.max(np.abs(spec)), np.max(np.abs(spec)))
            ax.plot(omega[::-1], np.ones_like(omega), '0.5', zorder=-3)
    else:
        lambdas = 2*np.pi*c/omega
        show_spectrum_complex(lambdas, spec, ax=ax, complex_valued=False, intensity=True)  

def compare_with_nareid(Z=10E-6, delta_z=10E-9, N=500, r=-1, freq=True):
    
    if freq:
        lambda_low = 350E-9
        omegas = np.linspace(2*np.pi*c/lambda_low, -2*np.pi*c/lambda_low, N)
        lambdas = 2*np.pi*c/omegas
    else:
        lambdas, omegas = generate_wavelengths(N=N, c=c)
    
    spectrum = generate_gaussian_spectrum(lambdas, mu=550E-9, sigma=30E-9)
   
    depths = generate_depths(delta_z=delta_z, max_depth=Z)
    lippmann, delta_intensity = lippmann_transform(lambdas, spectrum, depths, r=r)
        
    
    spectrum_replay_filt =  np.abs(apply_h(spectrum, Z, lambdas, lambdas, r=r))**2
    spectrum_replay_filt_nareid = np.abs( apply_h(spectrum, Z, lambdas, lambdas, nareid=True, r=r) )**2
    
    f, (ax1, ax2, ax3) = plt.subplots(3, figsize=(3.45, 1.5*3.45), sharex=True)
    plt_spectrum(omegas, spectrum, ax=ax1, x_label=False)
    plt_spectrum(omegas, spectrum_replay_filt, ax=ax2, x_label=False)
    plt_spectrum(omegas, spectrum_replay_filt_nareid, ax=ax3, x_label=True)
    ax1.set_ylabel('(a) Original')
    ax2.set_ylabel('(b) Reflection-based')
    ax3.set_ylabel('(c) Refraction-based')
    f.tight_layout(h_pad=-2.)
    f.subplots_adjust(hspace=-1.0)
    plt.savefig(fig_path + 'reflection_vs_refraction.pdf')
    
def compare_different_rhos(Z=10E-6, delta_z=10E-9, N=500, freq=True):
    
    if freq:
        lambda_low = 350E-9
        omegas = np.linspace(2*np.pi*c/lambda_low, -2*np.pi*c/lambda_low, N)
        lambdas = 2*np.pi*c/omegas
    else:
        lambdas, omegas = generate_wavelengths(N=N, c=c)
    
    spectrum = generate_gaussian_spectrum(lambdas, mu=550E-9, sigma=30E-9)   
    
    spectrum_r1 =  np.abs(apply_h(spectrum, Z, lambdas, lambdas, r=-1))**2
    spectrum_r05 =  np.abs(apply_h(spectrum, Z, lambdas, lambdas, r=-0.5))**2
    spectrum_r025 =  np.abs(apply_h(spectrum, Z, lambdas, lambdas, r=-0.25))**2
    
    
    f, (ax1, ax2, ax3) = plt.subplots(3, figsize=(3.45, 1.5*3.45), sharex=True)
    plt_spectrum(omegas, spectrum_r1, ax=ax1, x_label=False)
    plt_spectrum(omegas, spectrum_r05, ax=ax2, x_label=False)
    plt_spectrum(omegas, spectrum_r025, ax=ax3, x_label=True)
    ax1.set_ylabel('(a) r=-1')
    ax2.set_ylabel('(b) r=-0.5')
    ax3.set_ylabel('(c) r=-0.25')
    f.tight_layout(h_pad=-2.)
    f.subplots_adjust(hspace=-1.0)
    plt.savefig(fig_path + 'different_rhos.pdf')
   
def compare_infinite_depth(delta_z=10E-9, N=500, r=-1, freq=True):
    
    if freq:
        lambda_low = 350E-9
        omegas = np.linspace(2*np.pi*c/lambda_low, -2*np.pi*c/lambda_low, N)
        lambdas = 2*np.pi*c/omegas
    else:
        lambdas, omegas = generate_wavelengths(N=N, c=c)
    
    spectrum = generate_gaussian_spectrum(lambdas, mu=550E-9, sigma=30E-9)
    
    spectrum_replay_filt10 =  np.abs(apply_h(spectrum, 2.5E-6, lambdas, lambdas, r=r))**2
    spectrum_replay_filt20 =  np.abs(apply_h(spectrum, 10E-6, lambdas, lambdas, r=r))**2
    spectrum_replay_infniteZ =  np.abs( apply_h(spectrum, 5E-6, lambdas, lambdas, infinite=True, r=r) )**2

    f, (ax1, ax2, ax3) = plt.subplots(3, figsize=(3.45, 1.5*3.45), sharex=True)
    plt_spectrum(omegas, spectrum_replay_filt10, ax=ax1, x_label=False)
    plt_spectrum(omegas, spectrum_replay_filt20, ax=ax2, x_label=False)
    plt_spectrum(omegas, spectrum_replay_infniteZ, ax=ax3, x_label=True)
    ax1.set_ylabel('(a) $Z=2.5\mu m$')
    ax2.set_ylabel('(b) $Z=10\mu m$')
    ax3.set_ylabel('(c) Infinite depth')
    f.tight_layout(h_pad=-2.)
    f.subplots_adjust(hspace=-1.0)
    plt.savefig(fig_path + 'depth_experiment.pdf')
    
def compare_depths(delta_z=10E-9, N=500, r=-1, freq=True):
    
    if freq:
        lambda_low = 350E-9
        omegas = np.linspace(2*np.pi*c/lambda_low, -2*np.pi*c/lambda_low, N)
        lambdas = 2*np.pi*c/omegas
    else:
        lambdas, omegas = generate_wavelengths(N=N, c=c)
    
    spectrum_gauss = generate_gaussian_spectrum(lambdas, mu=550E-9, sigma=30E-9)
    spectrum_dirac = generate_mono_spectrum(lambdas, color=550E-9)
    
    gauss_replay_filt1 =  np.abs(apply_h(spectrum_gauss, 2.5E-6, lambdas, lambdas, r=r))**2
    gauss_replay_filt2 =  np.abs(apply_h(spectrum_gauss, 10E-6, lambdas, lambdas, r=r))**2
    gauss_replay_infniteZ =  np.abs( apply_h(spectrum_gauss, 5E-6, lambdas, lambdas, infinite=True, r=r) )**2
    
    dirac_replay_filt1 =  np.abs(apply_h(spectrum_dirac, 2.5E-6, lambdas, lambdas, r=r))**2
    dirac_replay_filt2 =  np.abs(apply_h(spectrum_dirac, 10E-6, lambdas, lambdas, r=r))**2
    dirac_replay_infniteZ =  np.abs(apply_h(spectrum_dirac, 5E-6, lambdas, lambdas, infinite=True, r=r))**2

    f, axes = plt.subplots(4,2, figsize=(3.45/0.6, 1.2*3.45/0.6), sharex='col')
    plt_spectrum(omegas, spectrum_gauss, ax=axes[0,0], x_label=False)
    plt_spectrum(omegas, gauss_replay_filt1, ax=axes[1,0], x_label=False)
    plt_spectrum(omegas, gauss_replay_filt2, ax=axes[2,0], x_label=False)
    plt_spectrum(omegas, gauss_replay_infniteZ, ax=axes[3,0], x_label=True)
    plt_spectrum(omegas, spectrum_dirac, ax=axes[0,1], x_label=False)
    plt_spectrum(omegas, dirac_replay_filt1, ax=axes[1,1], x_label=False)
    plt_spectrum(omegas, dirac_replay_filt2, ax=axes[2,1], x_label=False)
    plt_spectrum(omegas, dirac_replay_infniteZ, ax=axes[3,1], x_label=True)
#    vmax = 1.1*np.max(np.abs(dirac_replay_infniteZ[int(len(dirac_replay_infniteZ)*0.75):]))
#    plt_spectrum(omegas, dirac_replay_infniteZ, ax=axes[3,1], x_label=False, complex_valued=True, vmax=vmax, vmin=-vmax)
    axes[0,0].set_ylabel('(a) Original')
    axes[1,0].set_ylabel('(b) $Z=2.5~\mu m$')
    axes[2,0].set_ylabel('(c) $Z=10~\mu m$')
    axes[3,0].set_ylabel('(d) Infinite depth')
    axes[0,0].set_title('Gaussian spectrum')
    axes[0,1].set_title('Monochromatic spectrum')
    axes[3,0].set_xticks(np.linspace(-3E15, 3E15, 7))
    axes[3,1].set_xticks(np.linspace(-3E15, 3E15, 7))
    f.tight_layout(h_pad=-2.)
    f.subplots_adjust(hspace=-1.0)
    plt.savefig(fig_path + 'depth_experiment_2.pdf')
    
def compare_depthsPNAS(delta_z=10E-9, N=1000, r=-1, freq=True):
    
    if freq:
        lambda_low = 350E-9
        omegas = np.linspace(2*np.pi*c/lambda_low, -2*np.pi*c/lambda_low, N)
        lambdas = 2*np.pi*c/omegas
    else:
        lambdas, omegas = generate_wavelengths(N=N, c=c)
    
    spectrum_gauss = generate_gaussian_spectrum(lambdas, mu=550E-9, sigma=30E-9)
    
    gauss_replay_filt1 =  np.abs(apply_h(spectrum_gauss, 2E-6, lambdas, lambdas, r=r))**2
    gauss_replay_filt2 =  np.abs(apply_h(spectrum_gauss, 10E-6, lambdas, lambdas, r=r))**2
    gauss_replay_infniteZ =  np.abs( apply_h(spectrum_gauss, 5E-6, lambdas, lambdas, infinite=True, r=r) )**2
    
    f, axes = plt.subplots(4,1, figsize=(3.45*1.4, 1.7*3.45*1.4), sharex='col')
    plt_spectrum(omegas, spectrum_gauss, ax=axes[0], x_label=False)
    plt_spectrum(omegas, gauss_replay_filt1, ax=axes[1], x_label=False)
    plt_spectrum(omegas, gauss_replay_filt2, ax=axes[2], x_label=False)
    plt_spectrum(omegas, gauss_replay_infniteZ, ax=axes[3], x_label=True)
    axes[0].set_ylabel('(a) Original')
    axes[1].set_ylabel('(b) $Z=2~\mu m$')
    axes[2].set_ylabel('(c) $Z=10~\mu m$')
    axes[3].set_ylabel('(d) Infinite depth')
    axes[3].set_xticks(np.linspace(-3E15, 3E15, 7))
    f.tight_layout(h_pad=-2.)
    f.subplots_adjust(hspace=-1.0)
    plt.savefig(fig_path + 'depth_experiment_light.pdf')


def compare_skewing(Z=10E-6, delta_z=10E-9, N=500, freq=True):
    
    if freq:
        lambda_low = 350E-9
        omegas = np.linspace(2*np.pi*c/lambda_low, -2*np.pi*c/lambda_low, N)
        lambdas = 2*np.pi*c/omegas
    else:
        lambdas, omegas = generate_wavelengths(N=N, c=c)
    
    spectrum = generate_gaussian_spectrum(lambdas, mu=550E-9, sigma=30E-9)
    
    spectrum_replay_infniteZ =  apply_h(spectrum, 5E-6, lambdas, lambdas, infinite=True, r=-1)
    spectrum_replay_infniteZ_2 =  apply_h(spectrum, 5E-6, lambdas, lambdas, infinite=True, r=1)

    f, axes = plt.subplots(2,2, figsize=(3.45/0.6, 0.7*3.45/0.6), sharex='col')
    vmax = 1.1*np.max(np.abs(spectrum_replay_infniteZ[int(len(spectrum_replay_infniteZ)*0.75):]))
    plt_spectrum(omegas, spectrum_replay_infniteZ, ax=axes[0,0], x_label=False, complex_valued=True, vmax=vmax, vmin=-vmax)
    plt_spectrum(omegas, spectrum_replay_infniteZ_2, ax=axes[1,0], x_label=False, complex_valued=True, vmax=vmax, vmin=-vmax)
    axes[0,0].set_title('Reflected spectrum (complex)')
    axes[0,0].set_ylabel('(a) $r=-1$')
    axes[1,0].set_ylabel('(b) $r=1$')
    axes[1,0].set_xticks(np.linspace(-3E15, 3E15, 7))
    axes[1,0].set_xlabel('Frequency $\omega$')
   
    vmax = 1.1*np.max((np.abs(np.abs(spectrum_replay_infniteZ)**2)[int(len(spectrum_replay_infniteZ)*0.75):]))
    plt_spectrum(omegas, np.abs(spectrum_replay_infniteZ)**2, ax=axes[0,1], x_label=False, complex_valued=True, vmax=vmax, vmin=-vmax)
    plt_spectrum(omegas, np.abs(spectrum_replay_infniteZ_2)**2, ax=axes[1,1], x_label=False, complex_valued=True, vmax=vmax, vmin=-vmax)
    axes[0,1].set_title('Reflected spectrum (intensity)')
    axes[1,1].set_xticks(np.linspace(-3E15, 3E15, 7))
    axes[1,1].set_xlabel('Frequency $\omega$')
    f.tight_layout(h_pad=-2.)
    f.subplots_adjust(hspace=-1.0)
    plt.savefig(fig_path + 'skewing_both.pdf')

def compare_skewingPNAS(Z=10E-6, delta_z=10E-9, N=500, freq=True):
    
    if freq:
        lambda_low = 350E-9
        omegas = np.linspace(2*np.pi*c/lambda_low, -2*np.pi*c/lambda_low, N)
        lambdas = 2*np.pi*c/omegas
    else:
        lambdas, omegas = generate_wavelengths(N=N, c=c)
    
    spectrum = generate_gaussian_spectrum(lambdas, mu=550E-9, sigma=30E-9)
    
    spectrum_replay_infniteZ =  apply_h(spectrum, 5E-6, lambdas, lambdas, infinite=True, r=-1)
    spectrum_replay_infniteZ_2 =  apply_h(spectrum, 5E-6, lambdas, lambdas, infinite=True, r=1)

    f, axes = plt.subplots(2,1, figsize=(3.45*1.4, 0.7*3.45/0.6*1.4), sharex='col')
    vmax = 1.1*np.max(np.abs(spectrum_replay_infniteZ[int(len(spectrum_replay_infniteZ)*0.75):]))
    plt_spectrum(omegas, spectrum_replay_infniteZ, ax=axes[0], x_label=False, complex_valued=True, vmax=vmax, vmin=-vmax, col='0.')
    plt_spectrum(omegas, spectrum_replay_infniteZ_2, ax=axes[1], x_label=False, complex_valued=True, vmax=vmax, vmin=-vmax, col='0.')
    axes[0].plot(omegas, np.abs(spectrum_replay_infniteZ), zorder=-4)
    axes[0].fill_between(omegas, np.abs(spectrum_replay_infniteZ), -np.abs(spectrum_replay_infniteZ), color='#4C72B0', alpha=0.2, zorder=-5)
    axes[1].plot(omegas, np.abs(spectrum_replay_infniteZ_2), zorder=-4)
    axes[1].fill_between(omegas, np.abs(spectrum_replay_infniteZ_2), -np.abs(spectrum_replay_infniteZ_2), color='#4C72B0', alpha=0.2, zorder=-5)
    axes[0].set_ylabel('(a) $r=-1$')
    axes[1].set_ylabel('(b) $r=1$')
    axes[1].set_xticks(np.linspace(-3E15, 3E15, 7))
    axes[1].set_xlabel('Frequency $\omega$')
   
    vmax = 1.1*np.max((np.abs(np.abs(spectrum_replay_infniteZ)**2)[int(len(spectrum_replay_infniteZ)*0.75):]))
#    plt_spectrum(omegas, np.abs(spectrum_replay_infniteZ)**2, ax=axes[0,1], x_label=False, complex_valued=True, vmax=vmax, vmin=-vmax)
#    plt_spectrum(omegas, np.abs(spectrum_replay_infniteZ_2)**2, ax=axes[1,1], x_label=False, complex_valued=True, vmax=vmax, vmin=-vmax)
   
    plt.savefig(fig_path + 'skewing_PNAS.pdf')
        
def compare_with_filter(Z=10E-6, delta_z=10E-9, N=500, r=-1, freq=True):

    if freq:
        lambda_low = 350E-9
        omegas = np.linspace(2*np.pi*c/lambda_low, -2*np.pi*c/lambda_low, N)
        lambdas = 2*np.pi*c/omegas
    else:
        lambdas, omegas = generate_wavelengths(N=N, c=c)
    
#    spectrum = 3*generate_gaussian_spectrum(lambdas, mu=450E-9, sigma=30E-9) #+ 
    spectrum = generate_gaussian_spectrum(lambdas, mu=550E-9, sigma=30E-9)
#    spectrum = generate_mono_spectrum(lambdas, color=550E-9)
   
    depths = generate_depths(delta_z=delta_z, max_depth=Z)
    lippmann, delta_intensity = lippmann_transform(lambdas, spectrum, depths, r=r)
                    
    #window Lippmann
    L = int(len(lippmann)/1)
    ratio = 2
#    k = 9.7525E-9
    k = 0.03
    extinction = (ratio-1+np.cos( np.linspace(0, np.pi/2, L) ))/ratio
    extinction = np.exp(-2*k*depths/550E-9)
#    lippmann[-L:] *= extinction
        
    f, (ax1, ax2) = plt.subplots(1, 2)
    plt_spectrum(omegas, spectrum, ax=ax1)
    show_lippmann_transform(depths, lippmann, ax=ax2, black_theme=black_theme)
    
    spectrum_replay = inverse_lippmann(lippmann, lambdas, depths, return_intensity=True)
    
#    spectrum_replay_filt =  apply_h(spectrum, Z, lambdas, lambdas, r=r)
    spectrum_replay_filt =  apply_h(spectrum, Z, lambdas, lambdas, r=r)
    spectrum_replay_infniteZ =  np.abs( apply_h(spectrum, Z, lambdas, lambdas, infinite=False, r=r) )**2
    spectrum_replay_filt_nareid = np.abs( apply_h(spectrum, Z, lambdas, lambdas, nareid=True, r=r) )**2

    plt_spectrum(omegas, spectrum)
    plt.savefig(fig_path + 'original_freq.pdf')
    plt.title('original') 
    
    plt_spectrum(omegas, spectrum_replay)
    plt.title('lippmann') 
    
    plt_spectrum(omegas, spectrum_replay_filt)
#    plt_spectrum(omegas, spectrum_replay_filt, complex_valued=True)
    plt.savefig(fig_path + 'inverse_lippmann_freq.pdf')  
    plt.title('filter')   
    
    plt_spectrum(omegas, spectrum_replay_infniteZ)
    plt.savefig(fig_path + 'inverse_lippmann_infiniteZ_freq.pdf')  
    plt.title('filter infinite depth')
    
    plt_spectrum(omegas, spectrum_replay_infniteZ)
    plt_spectrum(omegas, spectrum_replay_filt, ax=plt.gca())
    plt.title('both finite/infinite')
    
    plt_spectrum(omegas, spectrum_replay_filt_nareid)
    plt.savefig(fig_path + 'inverse_nareid_freq.pdf')
    plt.title('nareid')
    
    f, (ax1, ax2, ax3, ax4) = plt.subplots(4, figsize=(3.45, 2*3.45), sharex=True)
    plt_spectrum(omegas, spectrum, ax=ax1, x_label=False)
#    plt_spectrum(omegas, spectrum_replay, ax=ax2, x_label=False)
    plt_spectrum(omegas, np.abs(spectrum_replay_filt)**2, ax=ax2, x_label=False)
    plt_spectrum(omegas, spectrum_replay_infniteZ, ax=ax3, x_label=False)
    plt_spectrum(omegas, spectrum_replay_filt_nareid, ax=ax4, x_label=True)
    ax1.set_ylabel('(a) Original')
    ax2.set_ylabel('(b) Reflection-based')
    ax3.set_ylabel('(c) Reflection-based (inf. depth)')
    ax4.set_ylabel('(d) Refraction-based')
    f.tight_layout(h_pad=-2.)
    f.subplots_adjust(hspace=-1.0)
    plt.savefig(fig_path + 'all_spectrums.pdf')
    
    f, (ax1, ax2, ax3, ax4) = plt.subplots(4, figsize=(3.45, 2*3.45), sharex=True)
    plt_spectrum(omegas, spectrum, ax=ax1, x_label=False)
#    plt_spectrum(omegas, spectrum_replay, ax=ax2, x_label=False)
    vmax = 1.1*np.max(np.abs(spectrum_replay_filt[int(len(spectrum_replay_filt)*0.75):]))
    plt_spectrum(omegas, spectrum_replay_filt, ax=ax2, x_label=False, complex_valued=True, vmax=vmax, vmin=-vmax)
    plt_spectrum(omegas, np.abs(spectrum_replay_filt)**2, ax=ax3, x_label=False)
    plt_spectrum(omegas, spectrum_replay_infniteZ, ax=ax4, x_label=True)
#    plt_spectrum(omegas, spectrum_replay_filt_nareid, ax=ax4, x_label=True)
    ax1.set_ylabel('(a) Original')
    ax2.set_ylabel('(b) Reflected (complex)')
    ax3.set_ylabel('(c) Reflected intensity')
    ax4.set_ylabel('(d) Reflected (inf. depth)')
    f.tight_layout(h_pad=-2.)
    f.subplots_adjust(hspace=-1.0)
    plt.savefig(fig_path + 'all_spectrums2.pdf')
    
    c1 = ct.from_xyz_to_rgb( ct.from_spectrum_to_xyz(lambdas, spectrum).reshape((1,1,-1)) )
    c2 = ct.from_xyz_to_rgb( ct.from_spectrum_to_xyz(lambdas, spectrum_replay).reshape((1,1,-1)) )
    c3 = ct.from_xyz_to_rgb( ct.from_spectrum_to_xyz(lambdas, spectrum_replay_infniteZ).reshape((1,1,-1)) )
    c4 = ct.from_xyz_to_rgb( ct.from_spectrum_to_xyz(lambdas, spectrum_replay_filt_nareid).reshape((1,1,-1)) )
    
    plt.figure();
    plt.imshow( np.r_[c1, c2, c3, c4])
    plt.savefig(fig_path + 'colors.pdf')
    
    
#    show_spectrum_frequencies(omegas, spectrum_replay/np.max(spectrum_replay))
#    show_spectrum_frequencies(omegas, spectrum_replay_filt/np.max(spectrum_replay_filt), ax=plt.gca())      
#    plt.title('both lippmann and filter') 
    
  


def analytic_vs_wave_transfer(Z=10E-6, delta_z=10E-9, N=500, lippmann_method=True, spectrum_shape='gauss'):
    
    N = 500
    if spectrum_shape == 'mono':
        epsilons = np.array([2.5E5, 5E5, 1E6, 2.5E6, 5E6])
        mus = np.array([0.01, 0.025, 0.05, 0.1, 0.2])*0.8
    else:
        epsilons = [1E6, 5E6, 1E7, 2E7]
        mus = np.array([0.025, 0.05, 0.1, 0.2])*2.5
        
    n0=1.45  
    labels = [r'(a) $\epsilon = 10^6$', r'(b) $\epsilon = 5 \cdot 10^6$', r'(c) $\epsilon = 10^7$', r'(d) $\epsilon = 2 \cdot 10^7$']
    
    lambdas, omegas = generate_wavelengths(N=N)
    
    if spectrum_shape == 'mono':
        spectrum = generate_mono_spectrum(lambdas, color=550E-9)
    else:
        spectrum = generate_gaussian_spectrum(lambdas, mu=550E-9, sigma=30E-9)
    
    depths = generate_depths(delta_z=delta_z, max_depth=Z)
    lippmann, delta_intensity = lippmann_transform(lambdas/n0, spectrum, depths)
    analytic_model = inverse_lippmann(lippmann, lambdas/n0, depths)
    
    ns = generate_lippmann_refraction_indices(delta_intensity, n0=n0, mu_n=0.01)
        
    f, axes = plt.subplots(1,4, figsize=(3.45/0.5, 3.45/4/0.5), sharex='col')
    
    for i, epsilon in enumerate(epsilons):  
        print(i)
        if lippmann_method:
            spectrum_reflected, _ = propagation_arbitrary_layers_Lippmann_spectrum(rs=lippmann/np.max(lippmann), d=delta_z, lambdas=lambdas, n0=n0, epsilon=epsilon, approximation=False)
        else:
            ns = generate_lippmann_refraction_indices(delta_intensity, n0=n0, mu_n=mus[i])
            spectrum_reflected, _ = propagation_arbitrary_layers_spectrum(ns, d=delta_z, lambdas=lambdas, symmetric=False)

#        spectrum_reflected /= np.max(spectrum_reflected)
        show_spectrum_complex(lambdas, spectrum_reflected/np.max(spectrum_reflected)*np.max(analytic_model), ax=axes[i], complex_valued=False, intensity=True, zorder=1, lw=1.5)
        axes[i].plot(lambdas*1E9, analytic_model, ':', zorder=2, c='0.7', lw=1.5)
        axes[i].set_xticks([400, 550, 700])
        axes[i].set_xticklabels([400, 'Wavelength (nm)', 700])
        axes[i].set_xlabel(labels[i])
              
    if lippmann_method:
        plt.savefig(fig_path + 'inverse_lippmann_saturation.pdf')  
    else:
        plt.savefig(fig_path + 'inverse_nareid_saturation.pdf')  
    
  
def strength_of_oscillations(Z=10E-6, delta_z=10E-9, N=500):
    
    lambdas, omegas = generate_wavelengths(N=N)
    spectrum = generate_gaussian_spectrum(lambdas, mu=550E-9, sigma=30E-9)
    depths = generate_depths(delta_z=delta_z, max_depth=Z)
    
    rs = [-0.1, -0.25, -0.5, -1]
    letters = ['(a)', '(b)', '(c)', '(d)']
        
    f, axes = plt.subplots(1,4, figsize=(3.45/0.5, 3.45/4/0.5), sharex='col')
    
    for i, r in enumerate(rs):  
        print(i)
        lippmann, _ = lippmann_transform(lambdas, spectrum, depths, r=r)
        replayed = inverse_lippmann(lippmann, lambdas, depths)
        
        show_spectrum_complex(lambdas, replayed, ax=axes[i], complex_valued=False, intensity=True, zorder=1, lw=1.5)
        axes[i].set_xticks([400, 550, 700])
        axes[i].set_xticklabels([400, 'Wavelength (nm)', 700])
#        axes[i].plot(lambdas*1E9, analytic_model, ':', zorder=2, c='0.7')
        label = letters[i] + ' $r = ' + str(r) + '$'
        axes[i].set_xlabel(label)
              
    plt.savefig(fig_path + 'strength_of_oscillations.pdf')  
    
    
def viewing_lighting_angles(Z=5E-6, delta_z=10E-9, N=500, r=-1):
    
    lambdas, omegas = generate_wavelengths(N=N)
    spectrum = generate_gaussian_spectrum(lambdas, mu=550E-9, sigma=30E-9)
    depths = generate_depths(delta_z=delta_z, max_depth=Z)
    
    
    alphas = [0, np.pi/8, np.pi/6, np.pi/4]
    alphas_str = ['0', r'\pi/8', r'\pi/6', r'\pi/4']
    letters = ['(a)', '(b)', '(c)', '(d)']
        
    f, axes = plt.subplots(len(alphas),len(alphas)+1, figsize=(3.45/0.5, 3.45/0.5*4/5), sharex='col')
    
    lippmann_0, _ = lippmann_transform(lambdas, spectrum, depths, r=r)
    spectrum_r = inverse_lippmann(lippmann_0, lambdas, depths)
    vmax = 1.1*np.max(spectrum_r)
    
    for i, alpha in enumerate(alphas):  
        print(i)
        lippmann, _ = lippmann_transform(lambdas, spectrum, depths*np.cos(alpha), r=r)
        
        show_lippmann_transform(depths, lippmann, ax=axes[i,0], black_theme=False, nolabel=True)
        
        for j, beta in enumerate(alphas):
        
            replayed = inverse_lippmann(lippmann, lambdas/np.cos(beta), depths)
#            replayed_i = inverse_lippmann(lippmann, lambdas, depths)
    #        replayed =  np.abs(apply_h(spectrum, Z, lambdas, lambdas/np.cos(alpha), r=r))**2
            
            show_spectrum(lambdas, replayed, ax=axes[i,j+1], show_background=True, vmax=vmax/(np.cos(alpha)**2), nolabel=True)
            
            axes[-1,j+1].set_xticks([400, 550, 700])
            axes[-1,j+1].set_xticklabels([400, '\lambda ~(nm)', 700])
            label = r' $\beta = ' + alphas_str[j] + '$'
            axes[0,j+1].set_title(label)
            
        label = r' $\alpha = ' + alphas_str[i] + '$'
        axes[i,0].set_ylabel(label)
            
    axes[-1,0].set_xticks([0, depths[-1]*1E6/2, depths[-1]*1E6])
    axes[-1,0].set_xticklabels([0, 'Depth (\mu m)', int(np.round(depths[-1]*1E6))])
        
#    f.tight_layout()
    f.subplots_adjust(hspace=-1.0)
              
    plt.savefig(fig_path + 'viewing_lighting_angles_.pdf')  
    
def experiment_hilbert(Z=10E-6, N=1000):
    
    lambdas, omegas = generate_wavelengths(N=N, c=c)  
    
    lambda_low = 350E-9
    omegas = np.linspace(-2*np.pi*c/lambda_low, 2*np.pi*c/lambda_low, N)
    lambdas = 2*np.pi*c/omegas     
    
    spectrum = generate_gaussian_spectrum(lambdas, mu=650E-9, sigma=30E-9) + generate_gaussian_spectrum(lambdas, mu=450E-9, sigma=20E-9)
           
    spectrum_replay_filt =  apply_h(spectrum, Z, lambdas, lambdas) 
    spectrum_replay_infniteZ =  apply_h(spectrum, Z, lambdas, lambdas, infinite=True) 
        
    show_spectrum_complex(omegas, spectrum, intensity=False)
    plt.title('spectrum original') 
    
    show_spectrum_complex(omegas, spectrum_replay_filt, intensity=False)
    plt.title('filtered')   
    
    show_spectrum_complex(omegas, spectrum_replay_infniteZ, intensity=False)
    plt.title('filtered infinite depth')
    
    show_spectrum_complex(omegas, spectrum_replay_infniteZ, intensity=False)
    show_spectrum_complex(omegas, spectrum_replay_filt, ax=plt.gca(), intensity=False)
    plt.legend(labels=['infinite', 'finite'])
    plt.title('both finite/infinite')
  
    
def different_reflectivity(Z=10E-6, delta_z=10E-9, N=500, spectrum_shape='gauss'):
        
    lambdas, omegas = generate_wavelengths(N=N, c=c)
    
    if spectrum_shape == 'mono':
        spectrum = generate_mono_spectrum(lambdas, color=550E-9)
    else:
        spectrum = generate_gaussian_spectrum(lambdas, mu=550E-9, sigma=30E-9)
    
    depths = generate_depths(delta_z=delta_z, max_depth=Z)
    
    plt.figure()
    
    rs = np.array([0.3, 0.5, 0.75, 1])
    
    for i, r in enumerate(rs):  
        print(i)
        
        lippmann, delta_intensity = lippmann_transform(lambdas, spectrum, depths, r=-r)
        spectrum_reflected = inverse_lippmann(lippmann, lambdas, depths)      
        
#        spectrum_reflected /= np.max(spectrum_reflected)
        show_spectrum_complex(lambdas, spectrum_reflected, ax=plt.gca(), complex_valued=False, intensity=True, zorder=0.5-i)
        fill_area_under(lambdas, spectrum_reflected, ax=plt.gca(), color=str(0.6 + 0.4*i/len(rs)), zorder=-i)
        
    plt.show()        
    plt.savefig(fig_path + 'inverse_lippmann_reflectance.pdf')  
 
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
    
if __name__ == '__main__':
    
    plt.close('all')
    
    Z = 5E-6
    N = 1000
    spectrum_shape = 'gauss'
        
    lambdas, omegas = generate_wavelengths(N=N, c=c)
#    lambdas*=1E9
    
#    h_filt = plot_h( lambdas, np.abs(h(lambdas, lambda_prime=550E-9, Z=Z))**2 )            
#    h_filt = plot_h( lambdas, h(lambdas, lambda_prime=550E-9, Z=Z) )
#    h_filt = plot_h( lambdas, c_inf(2*np.pi*c/550E-9 -omegas) )
#    h_filt = plot_h( lambdas, s_z_tilde(50E-6, 2*np.pi*c/550E-9 -omegas))
#    h_filt = plot_h( lambdas, s_z_tilde(500E-6, 2*np.pi*c/550E-9 -omegas))
    
#    h_filt_sym = plot_h( lambdas, np.abs(h_sym(lambdas, lambda_prime=550E-9, Z=Z))**2, name='h_sym', complex_valued=False )
#    h_filt_sym = plot_h( lambdas, h_sym(lambdas, lambda_prime=550E-9, Z=Z), name='h_sym', complex_valued=False )
#    plot_h( lambdas, np.abs(h(lambdas, lambda_prime=550E-9, Z=5E-6)), name='h (magnitude)', complex_valued=False)
#    plot_h( lambdas, np.abs(h_sym(lambdas, lambda_prime=550E-9, Z=5E-6)), name='h_sym (magnitude)', complex_valued=False)
    
#    spectrum = generate_gaussian_spectrum(lambdas)
#    spectrum = generate_mono_spectrum(lambdas, color=550E-9)
    
        
#    analytic_vs_wave_transfer(Z, N=N, lippmann_method=False, spectrum_shape=spectrum_shape)
#    analytic_vs_wave_transfer(Z, N=N, spectrum_shape=spectrum_shape)
#    
#    different_reflectivity(Z, N=N, spectrum_shape=spectrum_shape)
#    experiment_hilbert(Z, N)
#    spectrum_replay, H = apply_h_discrete(spectrum, Z, lambdas, symmetric=False)
    
    #plot filter
#    h_filt = plot_h( lambdas, s_z_tilde(Z, 2*np.pi*c/550E-9 -omegas) )
#    plt.savefig('s_z_tilde.pdf')
    
#    compare_with_lippmann(Z, N=N)
#    compare_with_filter(Z, N=N, r=0.7*np.exp(1j*np.deg2rad(-148)), freq=True)
#    compare_with_filter(Z, N=N, r=-1, freq=True)
#    compare_with_nareid(Z, N=N, r=-1, freq=True)
#    compare_different_rhos(Z, N=N, freq=True)
#    compare_infinite_depth(N=N, r=-1, freq=True)
#    compare_skewing(Z, N=N, freq=True)
    compare_skewingPNAS(Z, N=N, freq=True)
#    compare_depths(N=N, r=-1, freq=True)
#    compare_depthsPNAS(N=N, r=-1, freq=True)
#    strength_of_oscillations(N=N)
#    viewing_lighting_angles(N=N)



    
    