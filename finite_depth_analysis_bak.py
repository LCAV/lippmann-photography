# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 21:40:59 2017

@author: gbaechle
"""

import numpy as np
import scipy as sp
import scipy.interpolate
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches

from lippmann import *

import seaborn as sns
import seabornstyle as snsty

black_theme=False

snsty.setStyleMinorProject(black=black_theme)

sys.path.append("../")
import multilayer_optics_matrix_theory as matrix_theory

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

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
    y = np.where(x == 0, 1.0e-30, x) 
    
    return 1/y

    
def h(lambdas, lambda_prime, Z, r=-1, nareid=False):
    
    omega = 2*np.pi*c/lambdas 
    omega_prime = 2*np.pi*c/lambda_prime
    
    x = 2*omega_prime/c
    
#    return r/2*s_z_tilde(Z, omega_prime-omega)
    if nareid:
        return r/2*s_z_tilde(Z, omega_prime-omega) + np.conj(r)/2*s_z_tilde(Z, omega+omega_prime)
    else:
        return r/2*s_z_tilde(Z, omega_prime-omega) + (1+np.abs(r)**2)/2*s_z_tilde(Z, omega_prime) + np.conj(r)/2*s_z_tilde(Z, omega+omega_prime)
    
    numerator = 1j*c*np.exp(-1j*Z*x) * ((np.exp(1j*Z*x) -1 )*omega**2 + omega_prime**2 - omega_prime*(omega_prime*np.cos(2*Z*omega/c) + 1j*omega*np.sin(2*Z*omega/c)) )
    denominator = 2*(omega_prime**3 - omega**2*omega_prime)
    
    #fix the nans
    filt = numerator/denominator
        
    filt[denominator == 0] = -Z/2
        
#    show_spectrum_complex(lambdas, r/2*s_z_tilde(Z, omega_prime-omega) + (1+np.abs(r)**2)/2*s_z_tilde(Z, omega_prime) + np.conj(r)/2*s_z_tilde(Z, omega+omega_prime) ); plt.title('short')
#    show_spectrum_complex(lambdas, filt ); plt.title('exact') 
#    
#    show_spectrum_complex(lambdas,  r/2*s_z_tilde(Z, omega_prime-omega) + (1+np.abs(r)**2)/2*s_z_tilde(Z, omega_prime) + np.conj(r)/2*s_z_tilde(Z, omega+omega_prime)  )
#    show_spectrum_complex(lambdas, filt, ax=plt.gca() ); plt.title('both') 
            
    return filt
    
    
def h_sym(lambdas, lambda_prime, Z, r=-1):
    
    omega = 2*np.pi*c/lambdas 
    omega_prime = 2*np.pi*c/lambda_prime    
     
    return r/2*s_z(Z, omega-omega_prime) + (1+np.abs(r)**2)/2*s_z(Z, omega_prime) + np.conj(r)/2*s_z(Z, omega+omega_prime)                     
    
def plot_h(lambdas, filt, name='h', lambda_prime=500E-9, Z=5E-6, complex_valued=True, ax=None):
    
    ax = get_axis(ax)
    
    show_spectrum_complex(lambdas, filt, ax=ax, complex_valued=complex_valued)
    plt.savefig(fig_path + name + '.pdf') 
    
    return filt
    
def show_spectrum_frequencies(omegas, spectrum, ax=None, zorder=1):
        
    ax = get_axis(ax)
          
    vmax = 1.1*np.max(spectrum[int(len(spectrum)*0.75):])
    ax.set_ylim(0, vmax)          
    ax.plot(omegas, spectrum, lw=1)
    
    ax.set_xlim([np.min(omegas), np.max(omegas)])
    ax.set_yticks([])
  
    if black_theme:
        ax.set_xticks([])
    else:
#        ax.set_xticks([400, 500, 600, 700])
        ax.set_xlabel('Frequency $\omega$')
        
    _, omegas_visible = generate_wavelengths(1000, c=c)
    
    rect = patches.Rectangle((omegas_visible[-1], 0), omegas_visible[0]-omegas_visible[-1], vmax, facecolor='0.9', edgecolor='none', zorder=-10)

    ax.add_patch(rect)
    plot_gradient_line(omegas_visible, np.ones_like(omegas_visible)*vmax*1.05, ax=ax, lw=20)
            
    
    
def show_spectrum_complex(lambdas, spectrum, ax=None, complex_valued=True, true_spectrum=False, intensity=False, zorder=1):
    
    lam = lambdas*1E9
    
    ax = get_axis(ax)
    
    L = len(lam)
    
    if true_spectrum:
        cs = [wavelength_to_rgb(lam) for wavelength in lambdas]
    else:
        colors = plt.cm.Spectral_r(np.linspace(0, 1, L))
        cs = [colors[i] for i in range(L)]
        
    if complex_valued == True:
        if black_theme:
#            ax.scatter(lam, np.imag(spectrum), color='white', s=5, zorder=zorder)
            ax.plot(lam, np.imag(spectrum), color='white', lw=2, zorder=zorder-0.1)
        else:
#            ax.scatter(lam, np.imag(spectrum), color='0.5', s=5, zorder=zorder)
            ax.plot(lam, np.imag(spectrum), color='0.5', lw=2, zorder=zorder-0.1)
     
    if intensity:
        ax.set_ylim(0, 1.1*np.max(np.abs(spectrum)))
    else:
        ax.set_ylim(-1.1*np.max(np.abs(spectrum)), 1.1*np.max(np.abs(spectrum)))
        
    plot_gradient_line(lam, np.real(spectrum), ax=ax, zorder=zorder, lw=2, cs=cs)
#    ax.scatter(lam, np.real(spectrum), color=cs, s=3, zorder=zorder)
    ax.set_xlim([np.min(lam), np.max(lam)])
    ax.set_yticks([])
    if black_theme:
        ax.set_xticks([])
    else:
        ax.set_xticks([400, 500, 600, 700])
        ax.set_xlabel('Wavelength ($nm$)')
        
        
def plot_gradient_line(x, y, ax=None, zorder=1, lw=2, cs=None):
    
    L = len(x)    
    
    ax = get_axis(ax)
        
    if cs is None:
        cmap = plt.get_cmap('Spectral_r')
        colors = plt.cm.Spectral_r(np.linspace(0, 1, L))
        cs = [colors[i] for i in range(L)]
    else:
        cmap = LinearSegmentedColormap.from_list('test', cs)

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    lc = LineCollection(segments, cmap=cmap, norm=plt.Normalize(0, 1), zorder=zorder)
                                                
    lc.set_array(np.linspace(0, 1, L))
    lc.set_linewidth(lw)
    
    ax.add_collection(lc)
    ax.scatter(x, y, color=cs, s=3, zorder=zorder)
    
    
    
def show_sampling_lines(lambdas, spectrum, ax=None, zorder=-1, color='0.5'):
    
    ax = get_axis(ax)
    
    (markerlines, stemlines, baselines) = ax.stem(lambdas*1E9, spectrum, 'k-', markerfmt='k.', zorder=zorder)
    plt.setp(markerlines, color = 'black', markersize=2)
    plt.setp(stemlines, 'markersize', 0.5, color=color)
    plt.setp(baselines, 'linewidth', 0)
    
    
def fill_area_under(lambdas, spectrum, ax=None, zorder=0, color='0.5'):
    
    ax = get_axis(ax)
        
    ax.fill_between(lambdas*1E9, 0, spectrum, facecolor=color, zorder=zorder, interpolate=True)
    
    
def apply_h(spectrum, Z, lambdas, lambdas_prime, symmetric=False, nareid=False, infinite=False):
    
    new_spectrum = np.zeros(len(lambdas_prime), dtype=np.complex)
    
    omegas = 2*np.pi*c/lambdas 
    
    for i, lambda_prime in enumerate(lambdas_prime):
                
        if symmetric:
            new_spectrum[i] = -np.trapz(spectrum*h_sym(lambdas, lambda_prime, Z), omegas)
        else:
            new_spectrum[i] = -np.trapz(spectrum*h(lambdas, lambda_prime, Z, nareid=nareid), omegas)
    
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
    

def compare_with_filter(Z=10E-6, delta_z=10E-9, N=500):
    
#    lambdas, omegas = generate_wavelengths(N=N, c=c)
    
    lambda_low = 350E-9
    omegas = np.linspace(-2*np.pi*c/lambda_low, 2*np.pi*c/lambda_low, 10000)
    lambdas = 2*np.pi*c/omegas 
    
#    spectrum = 3*generate_gaussian_spectrum(lambdas, mu=450E-9, sigma=30E-9) #+ 
    spectrum = generate_gaussian_spectrum(lambdas, mu=550E-9, sigma=30E-9)
   
    depths = generate_depths(delta_z=delta_z, max_depth=Z)
    lippmann, delta_intensity = lippmann_transform(lambdas, spectrum, depths)
    
    spectrum_replay = inverse_lippmann(lippmann, lambdas, depths)
    
    spectrum_replay_filt = np.abs( apply_h(spectrum, Z, lambdas, lambdas) )**2
    spectrum_replay_filt_nareid = np.abs( apply_h(spectrum, Z, lambdas, lambdas, nareid=True) )**2

    show_spectrum_frequencies(omegas, spectrum)
    plt.savefig(fig_path + 'original_freq.pdf') 
    plt.title('original') 
    
    show_spectrum_frequencies(omegas, spectrum_replay)
    plt.title('lippmann') 
    
    show_spectrum_frequencies(omegas, spectrum_replay_filt)
    plt.savefig(fig_path + 'inverse_lippmann_freq.pdf')  
    plt.title('filter')
    
    show_spectrum_frequencies(omegas, spectrum_replay_filt_nareid)
    plt.savefig(fig_path + 'inverse_nareid_freq.pdf')  
    plt.title('nareid')
    
    show_spectrum_frequencies(omegas, spectrum_replay/np.max(spectrum_replay))
    show_spectrum_frequencies(omegas, spectrum_replay_filt/np.max(spectrum_replay_filt), ax=plt.gca())      
    
    plt.title('both') 
    
    print(spectrum_replay[:50])
    print(spectrum_replay_filt[:50])
    


def analytic_vs_wave_transfer(Z=10E-6, delta_z=10E-9, N=500, lippmann_method=True, spectrum_shape='gauss'):
    
    if spectrum_shape == 'mono':
        epsilons = np.array([2.5E5, 5E5, 1E6, 2.5E6, 5E6])
        mus = np.array([0.01, 0.025, 0.05, 0.1, 0.2])*0.8
    else:
        epsilons = [1E6, 2.5E6, 5E6, 1E7, 1.5E7]
        mus = np.array([0.01, 0.025, 0.05, 0.1, 0.2])*2.5
    
    n0=1.45  
    
    lambdas, omegas = generate_wavelengths(N=N)
    
    if spectrum_shape == 'mono':
        spectrum = generate_mono_spectrum(lambdas, color=550E-9)
    else:
        spectrum = generate_gaussian_spectrum(lambdas, mu=550E-9, sigma=30E-9)
    
    depths = generate_depths(delta_z=delta_z, max_depth=Z)
    lippmann, delta_intensity = lippmann_transform(lambdas/n0, spectrum, depths)
    
    ns = generate_lippmann_refraction_indices(delta_intensity, n0=n0, mu_n=0.01)
        
    plt.figure()
    
    for i, epsilon in enumerate(epsilons):  
        print(i)
        if lippmann_method:
            spectrum_reflected, _ = propagation_arbitrary_layers_Lippmann_spectrum(rs=lippmann/np.max(lippmann), d=delta_z, lambdas=lambdas, n0=n0, epsilon=epsilon, approximation=False)
        else:
            ns = generate_lippmann_refraction_indices(delta_intensity, n0=n0, mu_n=mus[i])
            spectrum_reflected, _ = propagation_arbitrary_layers_spectrum(ns, d=delta_z, lambdas=lambdas, symmetric=False)

#        spectrum_reflected /= np.max(spectrum_reflected)
        show_spectrum_complex(lambdas, spectrum_reflected, ax=plt.gca(), complex_valued=False, intensity=True, zorder=0.5-i)
        fill_area_under(lambdas, spectrum_reflected, ax=plt.gca(), color=str(0.6 + 0.4*i/len(epsilons)), zorder=-i)
        
    plt.show()        
    if lippmann_method:
        plt.savefig(fig_path + 'inverse_lippmann_saturation.pdf')  
    else:
        plt.savefig(fig_path + 'inverse_nareid_saturation.pdf')  
    
    
    
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
   
    
if __name__ == '__main__':
    
    Z = 5E-6
    N = 20000
    spectrum_shape = 'gauss'
        
    lambdas, omegas = generate_wavelengths(N=N, c=c)
#    lambdas*=1E9
    
#    h_filt = plot_h( lambdas, np.abs(h(lambdas, lambda_prime=550E-9, Z=Z))**2 )            
#    h_filt = plot_h( lambdas, h(lambdas, lambda_prime=550E-9, Z=Z) )
    h_filt = plot_h( lambdas, s_z_tilde(Z, 2*np.pi*c/550E-9 -omegas) )
    h_filt = plot_h( lambdas, s_z_tilde(50E-6, 2*np.pi*c/550E-9 -omegas))
    h_filt = plot_h( lambdas, s_z_tilde(500E-6, 2*np.pi*c/550E-9 -omegas))
    
#    h_filt_sym = plot_h( lambdas, np.abs(h_sym(lambdas, lambda_prime=550E-9, Z=Z))**2, name='h_sym', complex_valued=False )
#    h_filt_sym = plot_h( lambdas, h_sym(lambdas, lambda_prime=550E-9, Z=Z), name='h_sym', complex_valued=False )
#    plot_h( lambdas, np.abs(h(lambdas, lambda_prime=550E-9, Z=5E-6)), name='h (magnitude)', complex_valued=False)
#    plot_h( lambdas, np.abs(h_sym(lambdas, lambda_prime=550E-9, Z=5E-6)), name='h_sym (magnitude)', complex_valued=False)
    
#    spectrum = generate_gaussian_spectrum(lambdas)
#    spectrum = generate_mono_spectrum(lambdas, color=550E-9)
    
        
#    analytic_vs_wave_transfer(Z, N=N, lippmann_method=False, spectrum_shape=spectrum_shape)
#    analytic_vs_wave_transfer(Z, N=N, spectrum_shape=spectrum_shape)
    
#    different_reflectivity(Z, N=N, spectrum_shape=spectrum_shape)
    
#    compare_with_lippmann(Z, N=N)
    compare_with_filter(Z, N=N)
    
#    spectrum_replay, H = apply_h_discrete(spectrum, Z, lambdas, symmetric=False)


    
    