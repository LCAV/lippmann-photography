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
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit, line_search, least_squares

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches

from lippmann import *

import seabornstyle as snsty

black_theme=False

paper_mode = True

if paper_mode:
    snsty.setStyleMinorProject(black=black_theme)
    plt.rc('font', family='serif')
else:
    snsty.setStylePNAS()

plt.rc('text', usetex=True)

import multilayer_optics_matrix_theory as matrix_theory

c0 = 299792458
n0 = 1.5
c = c0/n0

def get_axis(ax=None):
    
    if ax is None:
        plt.figure()
        ax=plt.gca()
        
    return ax

def pad_values(x, padding, hold=False):
    delta = 0
    if hold is False:
        delta = x[1] - x[0]
        xm = x[0] + np.arange(-padding, 0)*delta
        xp = x[-1] + np.arange(1,padding+1)*delta
    else:
        xm = np.ones(padding)*x[0]
        xp = np.ones(padding)*x[-1]
                
    return np.r_[xm, x, xp]

def sinc_interp(x, s, u, padding=100):
   
#    Interpolates x, sampled at "s" instants
#    Output y is sampled at "u" instants ("u" for "upsampled") 
        
    if padding > 0 and s[1]-s[0]<0:
        return sinc_interp(x[::-1], s[::-1], u[::-1], padding)[::-1]
    
    elif padding > 0:
        u = pad_values(u, padding)
        s = pad_values(s, padding)
        x = pad_values(x, padding, hold=True)
        
   
    T = s[1] - s[0] # Find the period 
    
    sincM = np.tile(u, (len(s), 1)) - np.tile(s[:, None], (1, len(u)))
    
    y = x @ np.sinc(sincM/T)
    
    if padding > 0:
        return y[padding:-padding]
    else:
        return y


def s_z(Z, omega):
    """note: numpy sinc is sin(pi*x)/(pi*x)"""

    return Z*np.sinc(2*Z*omega/(np.pi*c))


def c_z(Z, omega):
    
    x = 2*Z*omega/c
    y = x + 1.0e-30

    return Z*(1-np.cos(y))/y    
    
    
def s_z_tilde(Z, omega):
    
    x = 2*Z*omega/c
    y = x + 1.0e-30
    
    return Z/(1j*y)*(1-np.exp(-1j*y))
#    return s_z(Z, omega) + 1j*c_z(Z, omega)


def s_z_tilde_dev(Z, omega, k0=0):
    
    x = c*k0+2j*omega*Z
    y = y = x + 1.0e-30
        
    return c*Z/y * (1-np.exp(-2j*omega*Z/c-k0))


def s_z_tilde_prime_Z(Z, omega, k0=0):
    return 2*Z*np.sinc(2*Z*omega/(np.pi*c))
#    return np.exp(-2j*omega*Z/c)
    
    
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
    return 1j/np.where(x == 0, 1j, x) 

    
def h(lambdas, lambda_prime, Z, r=-1, mode=1, k0=0):
    
    omega = 2*np.pi*c/lambdas 
    omega_prime = 2*np.pi*c/lambda_prime

    if mode == 2:
        s1 = s_z_tilde_dev(Z, omega_prime[:, None]-omega[None, :], k0=k0)
        s3 = s_z_tilde_dev(Z, omega_prime[:, None]+omega[None, :], k0=k0)
        return r/2*s1 + np.conj(r)/2*s3
    elif mode == 3:
        s2 = s_z_tilde_dev(Z, np.tile(omega_prime[:, None], (1, len(omega))), k0=k0)
        return (1+np.abs(r)**2)/2*s2
    else:
        s2 = s_z_tilde_dev(Z, omega_prime[:, None], k0=k0)
        s1 = s_z_tilde_dev(Z, omega_prime[:, None]-omega[None, :], k0=k0)
        s3 = s_z_tilde_dev(Z, omega_prime[:, None]+omega[None, :], k0=k0)
        return r/2*s1 + (1+np.abs(r)**2)/2*s2 + np.conj(r)/2*s3

    
def h_prime(lambdas, lambda_prime, Z, r=-1, k0=0):
    
    omega = 2*np.pi*c/lambdas 
    omega_prime = 2*np.pi*c/lambda_prime
    
    u = omega
    v = omega_prime
    
    return c/(u*(u-v)*(u+v))*((3*u**2-v**2)*np.sin(2*u*Z/c) + (3*u**2-v**2)*np.sin(2*(u-v)*Z/c) - 2*u*v*np.sin(2*v*Z/c) - 2*u*v*np.sin(4*v*Z/c) + 3*u**2*np.sin(2*(u+v)*Z/c) -v**2*np.sin(2*(u+v)*Z/c))


def h_sym(lambdas, lambda_prime, Z, r=-1):
    
    omega = 2*np.pi*c/lambdas 
    omega_prime = 2*np.pi*c/lambda_prime
     
    return r/2*s_z(Z, omega-omega_prime) + (1+np.abs(r)**2)/2*s_z(Z, omega_prime) + np.conj(r)/2*s_z(Z, omega+omega_prime)                     


def plot_h(lambdas, filt, name='h', lambda_prime=500E-9, Z=5E-6, complex_valued=True, ax=None, paper_mode=False):
    
    if paper_mode:
        plt.figure(figsize=(1.2*3.45, 1.2*3.45/2))
    else:
        plt.figure(figsize=(3.45*1.1, 0.6*3.45*1.1))
    
    ax = plt.gca()
    
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
    
    
def apply_h(spectrum, Z, lambdas, lambdas_prime, symmetric=False, infinite=False, r=-1, normalize=False, mode=1, k0=0):
    
    new_spectrum = np.zeros(len(lambdas_prime), dtype=np.complex)
    
    omegas = 2*np.pi*c/lambdas 
    omegas_prime = 2*np.pi*c/lambdas_prime 
        
    if infinite:
        spectrum_prime = np.interp(omegas[::-1], omegas[::-1], spectrum[::-1])[::-1] + np.interp(-omegas_prime[::-1], omegas[::-1], spectrum[::-1])[::-1]

        if symmetric:
            return spectrum_prime

    if infinite:

        for i, lambda_prime in enumerate(lambdas_prime):

            omega_prime = 2*np.pi*c/lambda_prime

            filt = r/2*c_high(omega_prime-omegas) + np.conj(r)/2*c_high(omegas+omega_prime)
            if omega_prime < 0:
                new_spectrum[i] = np.conj(r)*c*np.pi/4*spectrum_prime[i]+1j*np.trapz(spectrum*filt, omegas)
            else:
                new_spectrum[i] = r*c*np.pi/4*spectrum_prime[i]+1j*np.trapz(spectrum*filt, omegas)

        Pw = np.trapz(spectrum, omegas)
        new_spectrum += 1j*(1+np.abs(r)**2)/2*c_inf(omegas_prime)*Pw
                
    elif symmetric:
        new_spectrum = -np.trapz(spectrum*h_sym(lambdas, lambdas_prime, Z, r=r), omegas)
               
    else:
        new_spectrum = -np.trapz(spectrum*h(lambdas, lambdas_prime, Z, r=r, mode=mode, k0=k0), omegas)

    
    if normalize:
        return new_spectrum/np.trapz(spectrum, omegas)*(np.max(omegas)-np.min(omegas))
    else:
        return 2*new_spectrum
    
    
def compare_with_lippmann(Z=10E-6, delta_z=10E-9, N=500):
        
    lambdas_visible, omegas_visible = generate_wavelengths(N=N, c=c)
    lambdas, omegas = generate_wavelengths_sinc(Z=Z, c=c)
    
    spectrum = generate_gaussian_spectrum(lambdas_visible, mu=550E-9, sigma=30E-9) #+ generate_gaussian_spectrum(lambdas_visible, mu=650E-9, sigma=20E-9)
#    spectrum = generate_mono_spectrum(lambdas_visible)
#    spectrum = generate_rect_spectrum(lambdas_visible)
    
    depths = generate_depths(delta_z=delta_z, max_depth=Z)
    lippmann, delta_intensity = lippmann_transform(lambdas_visible, spectrum, depths)
        
    spectrum_replay = np.abs(apply_h(spectrum, Z, lambdas_visible, lambdas))**2
    spectrum_replay_visible = np.abs(apply_h(spectrum, Z, lambdas_visible, lambdas_visible))**2
    
    spectrum_replay_nareid = np.abs(apply_h(spectrum, Z, lambdas_visible, lambdas_visible, mode=2))**2

    spectrum_interpolated = sp.interpolate.interp1d(omegas, spectrum_replay, kind='cubic', bounds_error=False, fill_value='extrapolate')(omegas_visible)
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

def compare_with_nareid(Z=5E-6, delta_z=10E-9, N=500, r=-1, freq=True):
    
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
    spectrum_replay_filt_nareid = np.abs( apply_h(spectrum, Z, lambdas, lambdas, mode=2, r=r) )**2
    
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
    
#    f, (ax1, ax2) = plt.subplots(1,2, figsize=(1.5*3.45, 1.5*0.5*3.45))
#    f, (ax1, ax2) = plt.subplots(1,2, figsize=(3.45/0.55, 0.5*3.45/0.55))
    f, (ax1, ax2) = plt.subplots(2, figsize=(3.45*1.1, 3.45*1.1), sharex='col')
    
   
    plt_spectrum(omegas, spectrum_replay_filt, ax=ax1, x_label=False)
    plt_spectrum(omegas, spectrum_replay_filt_nareid, ax=ax2, x_label=True)
    ax1.set_ylabel('(a) Reflection-based')
    ax2.set_ylabel('(b) Refraction-based')
    f.tight_layout(h_pad=-2.)
    f.subplots_adjust(hspace=-1.0)
    plt.savefig(fig_path + 'reflection_vs_refraction_paper.pdf')
    
    
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
    
def compare_infinite_depth_PNAS(delta_z=10E-9, N=2000, r=-1, freq=True):
    
    if freq:
        lambda_low = 350E-9
        omegas = np.linspace(2*np.pi*c/lambda_low, -2*np.pi*c/lambda_low, N)
        lambdas = 2*np.pi*c/omegas
    else:
        lambdas, omegas = generate_wavelengths(N=N, c=c)
    
    spectrum = generate_gaussian_spectrum(lambdas, mu=550E-9, sigma=30E-9)
    
    spectrum_replay_filt10 =  np.abs(apply_h(spectrum, 2.5E-6, lambdas, lambdas, r=r))**2
    spectrum_replay_filt20 =  np.abs(apply_h(spectrum, 10E-6, lambdas, lambdas, r=r))**2
    spectrum_replay_filt20_win =  np.abs(apply_h(spectrum, 10E-6, lambdas, lambdas, r=r, k0=2))**2
    spectrum_replay_infniteZ =  np.abs( apply_h(spectrum, 5E-6, lambdas, lambdas, infinite=True, r=r) )**2

    f, axes = plt.subplots(5, 2, figsize=(1.2*3.45, 1.2*1.5*3.45), sharex='col')
    
    depths1 = np.linspace(0,2.5E-6,N)
    interf, _ = lippmann_transform(lambdas, spectrum, depths1, r=r, k0=0); interf[-1] = 0
    show_lippmann_transform(depths1, interf, ax=axes[1,0], nolabel=True)
    
    depths2 = np.linspace(0,10E-6,N)
    interf, _ = lippmann_transform(lambdas, spectrum, depths2, r=r, k0=0); interf[-1] = 0
    show_lippmann_transform(depths2, interf, ax=axes[2,0], nolabel=True)
    
    interf, _ = lippmann_transform(lambdas, spectrum, depths2, r=r, k0=2); interf[-1] = 0
    show_lippmann_transform(depths2, interf, ax=axes[3,0], nolabel=True)
    
    depths3 = np.linspace(0,12E-6,N)
    interf, _ = lippmann_transform(lambdas, spectrum, depths3, r=r, k0=0)
    show_lippmann_transform(depths3, interf, ax=axes[4,0], nolabel=False)
    
    show_spectrum(lambdas, spectrum,  ax=axes[0,1], visible=True, nolabel=True)
    show_spectrum(lambdas, spectrum_replay_filt10,  ax=axes[1,1], visible=True, nolabel=True)
    show_spectrum(lambdas, spectrum_replay_filt20,  ax=axes[2,1], visible=True, nolabel=True)
    show_spectrum(lambdas, spectrum_replay_filt20_win,  ax=axes[3,1], visible=True, nolabel=True)
    show_spectrum(lambdas, spectrum_replay_infniteZ,  ax=axes[4,1], visible=True, nolabel=False)
    
    axes[0,0].axis('off')
    axes[0,0].set_ylabel('(a) Original')
    axes[1,0].set_ylabel('(b) $Z=2.5\mu m$')
    axes[2,0].set_ylabel('(c) $Z=10\mu m$')
    axes[3,0].set_ylabel('(d) Damped')
    axes[4,0].set_ylabel('(e) Inf. $Z$')
    f.tight_layout(h_pad=-2.)
    f.subplots_adjust(hspace=-1.0)
    plt.savefig(fig_path + 'depth_experiment_PNAS.pdf')
    
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

#    f, axes = plt.subplots(4,2, figsize=(3.45/0.55, 1.2*3.45/0.55), sharex='col')
    f, axes = plt.subplots(4,2, figsize=(3.45/0.45, 1.15*3.45/0.45), sharex='col')
    
    
    plt_spectrum(omegas, spectrum_gauss, ax=axes[0,1], x_label=False)
    plt_spectrum(omegas, gauss_replay_filt1, ax=axes[1,1], x_label=False)
    plt_spectrum(omegas, gauss_replay_filt2, ax=axes[2,1], x_label=False)
    plt_spectrum(omegas, gauss_replay_infniteZ, ax=axes[3,1], x_label=True)
    plt_spectrum(omegas, spectrum_dirac, ax=axes[0,0], x_label=False)
    plt_spectrum(omegas, dirac_replay_filt1, ax=axes[1,0], x_label=False)
    plt_spectrum(omegas, dirac_replay_filt2, ax=axes[2,0], x_label=False)
    plt_spectrum(omegas, dirac_replay_infniteZ, ax=axes[3,0], x_label=True)
#    vmax = 1.1*np.max(np.abs(dirac_replay_infniteZ[int(len(dirac_replay_infniteZ)*0.75):]))
#    plt_spectrum(omegas, dirac_replay_infniteZ, ax=axes[3,1], x_label=False, complex_valued=True, vmax=vmax, vmin=-vmax)
    axes[0,0].set_ylabel('(a) Original')
    axes[1,0].set_ylabel('(b) $Z=2.5~\mu m$')
    axes[2,0].set_ylabel('(c) $Z=10~\mu m$')
    axes[3,0].set_ylabel('(d) Infinite depth')
    axes[0,1].set_title('Gaussian spectrum')
    axes[0,0].set_title('Monochromatic spectrum')
    axes[3,0].set_xticks(np.linspace(-3E15, 3E15, 7))
    axes[3,1].set_xticks(np.linspace(-3E15, 3E15, 7))
    f.tight_layout(h_pad=-2.)
    f.subplots_adjust(hspace=-1.0)
    plt.savefig(fig_path + 'depth_experiment_2.pdf')
    
def compare_depthsPNAS(N=1000, r=-1, freq=True):
    
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
    
def compare_skewingPNAS2(Z=10E-6, delta_z=10E-9, N=500, one_sided=True):
    
    lambda_low = 350E-9
    if one_sided:
        omegas = np.linspace(2*np.pi*c/lambda_low, 0, N)
    else:
        omegas = np.linspace(2*np.pi*c/lambda_low, -2*np.pi*c/lambda_low, N)
    lambdas = 2*np.pi*c/omegas

    lambdas_v, omegas_v = generate_wavelengths(N=N, c=c)
    
    spectrum = generate_gaussian_spectrum(lambdas, mu=550E-9, sigma=30E-9)
    spectrum_v = generate_gaussian_spectrum(lambdas_v, mu=550E-9, sigma=30E-9)
    
    spectrum_replay_infniteZ =  apply_h(spectrum, 5E-6, lambdas, lambdas, infinite=True, r=-1)
    spectrum_replay_infniteZ_2 =  apply_h(spectrum, 5E-6, lambdas, lambdas, infinite=True, r=1)

    f, axes = plt.subplots(2,2, figsize=(0.6*2*3.45*1.4, 0.6*0.7*3.45/0.6*1.4), sharex='col')
    
    depths = np.linspace(0,3E-6,N)
    interf1, _ = lippmann_transform(lambdas_v, spectrum_v, depths, r=-1, k0=0)
    show_lippmann_transform(depths, interf1, ax=axes[0,0], nolabel=True)
    interf2, _ = lippmann_transform(lambdas_v, spectrum_v, depths, r=1, k0=0)
    show_lippmann_transform(depths, interf2, ax=axes[1,0], nolabel=True)
    
    
    if one_sided:
        vmax = 1.1*np.max(np.abs(spectrum_replay_infniteZ[:int(len(spectrum_replay_infniteZ)*0.5)]))
        axes[1,1].set_xticks(np.linspace(0, 3E15, 4))
    else:
        vmax = 1.1*np.max(np.abs(spectrum_replay_infniteZ[int(len(spectrum_replay_infniteZ)*0.75):]))
        axes[1,1].set_xticks(np.linspace(-3E15, 3E15, 7))
        
    plt_spectrum(omegas, spectrum_replay_infniteZ, ax=axes[0,1], x_label=False, complex_valued=True, vmax=vmax, vmin=-vmax, col='0.')
    plt_spectrum(omegas, spectrum_replay_infniteZ_2, ax=axes[1,1], x_label=False, complex_valued=True, vmax=vmax, vmin=-vmax, col='0.')
    axes[0,1].plot(omegas, np.abs(spectrum_replay_infniteZ), zorder=-4)
    axes[0,1].fill_between(omegas, np.abs(spectrum_replay_infniteZ), -np.abs(spectrum_replay_infniteZ), color='#4C72B0', alpha=0.2, zorder=-5)
    axes[1,1].plot(omegas, np.abs(spectrum_replay_infniteZ_2), zorder=-4)
    axes[1,1].fill_between(omegas, np.abs(spectrum_replay_infniteZ_2), -np.abs(spectrum_replay_infniteZ_2), color='#4C72B0', alpha=0.2, zorder=-5)
    axes[0,0].set_ylabel('(a1) $r=-1$')
    axes[1,0].set_ylabel('(a2) $r=1$')    
    axes[1,1].set_xlabel('Frequency $\omega$')
    
    axes[0,0].set_yticks([0])
    axes[1,0].set_yticks([0])
    axes[1,0].set_xticks(np.arange(4))
    axes[1,0].set_xlabel('Depth $(\mu m)$')
   
    plt.savefig(fig_path + 'skewing_PNAS2.pdf')
    

def plot_reflecting_wave(N, r, A0=1, N_time=10, ax=None):

    time = np.linspace(0,1, N_time)
    space = (np.arange(N)/(N)*6*np.pi)
    omega = 2*np.pi/5
    
    if ax is None:
        plt.figure()
        ax = plt.gca()
    
    for t in time:
        wave = A0*np.exp(-1j*space)*np.exp(1j*omega*t)
        ax.plot(space, np.real(wave))
        
    for t in time[::-1]:
        wave = r*A0*np.exp(1j*space)*np.exp(1j*omega*t)
        ax.plot(space, np.real(wave))
        
def plot_sum_waves(N, r, A0=1, N_time=10, ax=None):

    time = np.linspace(0,1, N_time)
    space = (np.arange(N)/(N)*6*np.pi)
    omega = 2*np.pi/5
    
    if ax is None:
        plt.figure()
        ax = plt.gca()
    
    for t in time:
        wave1 = A0*np.exp(-1j*space)*np.exp(1j*omega*t)
        wave2 = r*A0*np.exp(1j*space)*np.exp(1j*omega*t)
        sum_waves = wave1 + wave2
        ax.plot(space, np.real(sum_waves))
        
    ax.set_ylim([-2.2,2.2])
    
def plot_standing_wave_examples():
    N_time = 5
    r = 1
    with sns.color_palette("RdBu_r", 2*N_time):
        
        f, axes = plt.subplots(2,2, figsize=(2*3.45, 3.45), sharex=True)
        
        plot_reflecting_wave(N, N_time=N_time, r=-1, ax=axes[0,0])
        plot_sum_waves(N, N_time=N_time, r=-1, ax=axes[1,0])
        
        plot_reflecting_wave(N, N_time=N_time, r=1, ax=axes[0,1])
        plot_sum_waves(N, N_time=N_time, r=1, ax=axes[1,1])
        
        remove_ticklabels(axes[0,0])
        remove_ticklabels(axes[0,1])
        remove_ticklabels(axes[1,0])
        remove_ticklabels(axes[1,1])
        
        axes[0,0].set_ylabel('Forward/backward waves')
        axes[1,0].set_ylabel('Sum of 2 waves')
        
        axes[0,0].set_title('$r=-1$')
        axes[0,1].set_title('$r=1$')
    
    plt.savefig(fig_path + 'standing_waves.pdf')
    
    
def remove_ticklabels(ax):
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    
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
    spectrum_replay_filt_nareid = np.abs( apply_h(spectrum, Z, lambdas, lambdas, mode=2, r=r) )**2

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
        
#    f, axes = plt.subplots(1,4, figsize=(3.45/0.5, 3.45/4/0.5), sharex='col')
    
    f, axes = plt.subplots(2,2, figsize=(3.45*1.1, 3.45*1.1), sharex='col')
    
    axes = axes.flatten()
    
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
        
    f, axes = plt.subplots(len(alphas),len(alphas)+1, figsize=(3.45/0.5, 3.45/0.5*4/5), sharex='col')
    
    #determine the scale
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
            axes[-1,j+1].set_xticklabels([400, '$\lambda ~(nm)$', 700])
            label = r' $\beta = ' + alphas_str[j] + '$'
            axes[0,j+1].set_title(label)
            
        label = r' $\alpha = ' + alphas_str[i] + '$'
        axes[i,0].set_ylabel(label)
            
    axes[-1,0].set_xticks([0, depths[-1]*1E6/2, depths[-1]*1E6])
    axes[-1,0].set_xticklabels([0, 'Depth ($\mu m$)', int(np.round(depths[-1]*1E6))])
        
    f.tight_layout()
#    f.subplots_adjust(hspace=-1)
             
    plt.savefig(fig_path + 'viewing_lighting_angles_.pdf')  
    
    
def experiment_skewing(path, visible=True):
    
    f, axes = plt.subplots(5, 3, figsize=(3.45/0.5, 3.45/0.5), sharex='col')
    
    for i, wavelength in enumerate(['450', '500', '550', '600', '650']):
        for j, (file_name, exp) in enumerate(zip(['gt', 'hg', 'air'], ['512', '128', '512'])):
            lambdas, spectrum = load_lippmann(path + file_name + '_' + wavelength + '_' + exp)
            if i == 4:
                show_spectrum(lambdas, spectrum, ax=axes[i,j], visible=visible, show_background=True, nolabel=False)
            else:
                show_spectrum(lambdas, spectrum, ax=axes[i,j], visible=visible, show_background=True, nolabel=True)
            
    axes[0,0].set_title('Original spectrum')
    axes[0,1].set_title('Hg reflector')
    axes[0,2].set_title('Air reflector')
    
    plt.savefig(fig_path + 'skewing_experiment.pdf')
    
    
def experiment_holo_vs_lipp(path, visible=True):
    
#    f, axes = plt.subplots(1, 2, figsize=(3.45/0.5*0.6, 3.45*0.6), sharex='col')
    f, axes = plt.subplots(1, 2, figsize=(3.45*1.6/1.3, 3.45/1.2/1.3), sharex='col')
#    figsize=(3.45/0.5, 3.45/4/0.5)
    
    lambdas, spectrum = load_lippmann(path + 'air_550_512')
    show_spectrum(lambdas, spectrum, ax=axes[0], visible=visible, show_background=True, nolabel=False)
    
    lambdas, spectrum = load_lippmann(path + 'holo_64')
    show_spectrum(lambdas, spectrum, ax=axes[1], visible=visible, show_background=True, nolabel=False)
    
    plt.savefig(fig_path + 'lipp_vs_holo.pdf') 

    
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


def get_vmax(values, freq):
    
    if freq:
        return 1.1*np.max(np.abs(values[int(len(values)*0.75):]))
    else:
        return 1.1*np.max(np.abs(values))



if __name__ == '__main__':
    
    plt.close('all')
    
    Z = 12.7E-6
    N = 500
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
    
    #plot filters
#    h_filt = plot_h( lambdas, s_z_tilde(5E-6, 2*np.pi*c/550E-9 -omegas), paper_mode=paper_mode)
#    plt.savefig('s_z_tilde.pdf')
##    
#    h_filt = plot_h( lambdas, s_z_tilde_dev(5E-6, 2*np.pi*c/550E-9 -omegas, k0=2), paper_mode=paper_mode)
#    plt.savefig('s_z_tilde_dev.pdf')
    
#    compare_with_lippmann(Z, N=N)
#    compare_with_filter(Z, N=N, r=0.7*np.exp(1j*np.deg2rad(-148)), freq=True)
#    compare_with_filter(Z, N=N, r=-1, freq=True)
#    compare_with_nareid(Z=5E-6, N=N, r=-1, freq=True)
#    compare_different_rhos(Z, N=N, freq=True)
#    compare_infinite_depth(N=N, r=-1, freq=True)
#    compare_infinite_depth_PNAS(N=2000, r=-1, freq=True)
#    compare_skewing(Z, N=N, freq=True)
#    compare_skewingPNAS(Z, N=N, freq=True)
    compare_skewingPNAS2(Z, N=N)
#    compare_depths(N=N, r=-1, freq=True)
#    compare_depthsPNAS(N=N, r=-1, freq=True)
#    strength_of_oscillations(N=N)
#    viewing_lighting_angles(N=N)
    
#    experiment_skewing(path='/Users/gbaechle/EPFL/PhD/BRDF Data and Rendering Engines/Goniophotometer/2018-07-12 air_hg/')
#    experiment_holo_vs_lipp(path='/Users/gbaechle/EPFL/PhD/BRDF Data and Rendering Engines/Goniophotometer/2018-07-13 lipp_vs_holo/')

#    plot_standing_wave_examples()
        
    
    
    
