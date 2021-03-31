#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 19:56:16 2018

@author: gbaechle
"""

import numpy as np
import scipy as sp
import scipy.stats
from scipy.special import erfc
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import least_squares

import seabornstyle as snsty

import sys
sys.path.append("../")
from lippmann import *
#from finite_depth_analysis import *

plt.rc('text', usetex=True)
plt.rc('font', family='serif')


plt.close('all')
fig_path = 'PNAS/'


def plot_pipeline(N=500):
    
    Z = 5E-6
    r = 0.8
    t = 0.
    
    lambdas, omegas = generate_wavelengths(N) #3000
    depths = np.linspace(0,Z*(1-1/N),N)
    #    depths = generate_depths(max_depth=5E-6)
    omegas = 2 * np.pi * c / lambdas 
    
    spectrum = generate_gaussian_spectrum(lambdas=lambdas, mu=530E-9, sigma=20E-9) + generate_gaussian_spectrum(lambdas=lambdas, mu=460E-9, sigma=40E-9) + 0.7*generate_gaussian_spectrum(lambdas=lambdas, mu=650E-9, sigma=55E-9)    
    spectrum = generate_mono_spectrum(lambdas=lambdas, color=530E-9)  
    lippmann_complex = lippmann_transform_complex(lambdas, spectrum, depths, r=r, t=t)
    lippmann_no_window = lippmann_transform(lambdas, spectrum, depths, r=r)[0] 
    
    window = (np.cos(np.linspace(0, np.pi, len(depths)))+1)/2
#    window = np.linspace(1,0, len(depths))
#    window = np.exp(np.linspace(0, -7, len(depths)))
    window = erfc(np.linspace(0,0.5,len(depths))) 
#    window = np.ones(len(depths))
    lippmann = lippmann_no_window*window
    
    spectrum_reconstructed_complex = inverse_lippmann(lippmann, lambdas, depths, return_intensity=False)
    spectrum_reconstructed = np.abs(spectrum_reconstructed_complex)**2
    
    
    plt.figure(figsize=(3.42/3, 3.42/3))
    show_spectrum(lambdas, spectrum, ax=plt.gca()) 
    plt.gca().set_yticks([0])
    plt.savefig(fig_path + 'original.pdf')
    plt.title('original spectrum') 
    
    plt.figure(figsize=(3.42/3, 3.42/3))
    show_lippmann_transform(depths, lippmann_complex, ax=plt.gca(), complex_valued=True)
    plt.gca().axhline(y=0, color='k', zorder=-10, lw=0.5)
    plt.savefig(fig_path + 'lippmann_complex.pdf')
    plt.title('Lippmann transform (complex)') 
    
    plt.figure(figsize=(3.42/3, 3.42/3))
    show_lippmann_transform(depths, lippmann_no_window, ax=plt.gca()) 
    plt.gca().set_yticks([0])
    plt.savefig(fig_path + 'plate_density.pdf')
    plt.title('Lippmann transform')
    
    plt.figure(figsize=(3.42/3, 3.42/3))
    show_lippmann_transform(depths, lippmann, ax=plt.gca()) 
    plt.gca().set_yticks([0])
    plt.savefig(fig_path + 'plate_density_windowed.pdf')
    plt.title('Lippmann transform  windowed')
    
    plt.figure(figsize=(3.42/3, 3.42/3))
    show_spectrum(lambdas, np.real(spectrum_reconstructed_complex), ax=plt.gca())
    plt.gca().plot(lambdas*1E9, np.imag(spectrum_reconstructed_complex), c='0.7', zorder=-2)
    plt.gca().set_ylim(-1.1*np.max(np.abs(spectrum_reconstructed_complex)), 1.1*np.max(np.abs(spectrum_reconstructed_complex)))
    plt.gca().axhline(y=0, color='k', zorder=-10, lw=0.5)
    plt.gca().set_yticks([0])
    plt.savefig(fig_path + 'replay_complex.pdf')
    plt.title('spectrum replayed (complex)') 
    
    plt.figure(figsize=(3.42/3, 3.42/3))
    show_spectrum(lambdas, spectrum_reconstructed, ax=plt.gca()) 
    plt.gca().set_yticks([0])
    plt.savefig(fig_path + 'replay.pdf')
    plt.title('spectrum replayed') 
    
    f, axes = plt.subplots(1, 5, figsize=(3.45/0.5*1.4, 3.45/4.6/0.5*1.4))
    show_spectrum(lambdas, spectrum, ax=axes[0], short_display=True)
    axes[0].set_yticks([0])
#    axes[0].set_xticklabels([400, '$\lambda~(nm)$', 700])
    show_lippmann_transform(depths, lippmann_no_window, ax=axes[1], short_display=True)
    axes[1].axhline(y=0, color='k', zorder=-10, lw=0.5)
    show_lippmann_transform(depths, lippmann, ax=axes[2], short_display=True)
    axes[2].axhline(y=0, color='k', zorder=-10, lw=0.5)
    show_spectrum(lambdas, np.real(spectrum_reconstructed_complex), ax=axes[3], short_display=True)
    axes[3].plot(lambdas*1E9, np.imag(spectrum_reconstructed_complex), c='0.7', zorder=-2)
    axes[3].set_ylim(-1.1*np.max(np.abs(spectrum_reconstructed_complex)), 1.1*np.max(np.abs(spectrum_reconstructed_complex)))
    axes[3].axhline(y=0, color='k', zorder=-10, lw=0.5)
    axes[3].set_yticks([0])
#    axes[3].set_xticklabels([400, '$\lambda~(nm)$', 700])
    show_spectrum(lambdas, spectrum_reconstructed, ax=axes[4], short_display=True) 
    axes[4].set_yticks([0])
#    axes[4].set_xticklabels([400, '$\lambda~(nm)$', 700])
    
    axes[0].set_title('(a) Original spectrum')
    axes[1].set_title('(b) Intensity of interferences')
    axes[2].set_title('(c) Silver density')
    axes[3].set_title('(d) Complex wavefunction')
    axes[4].set_title('(e) Replayed intensity')
    plt.savefig(fig_path + 'pipeline.pdf')
    
    
    
def plot_sinewaves(N=500, periods=3):
    
    t = np.linspace(0,2*periods*np.pi,N)
    t2 = np.linspace(0,2*periods*np.pi,10*N)
    wave = np.sin(t)
    
    plt.figure(figsize=(10,2))
    plt.plot(t, wave, color='#4FAADF')
    plt.ylim([-1.2, 1.2])
    plt.savefig('scalar_field.pdf')
    
    plt.figure(figsize=(10, 2))
    stem_with_color(t, wave-0.05, plt.gca(), color='#4FAADF', pos=True)
    stem_with_color(t, wave+0.05, plt.gca(), color='#4FAADF', pos=False)
    plt.plot(t2, np.sin(t2), color='k')           
    plt.ylim([-1.2, 1.2])
    plt.savefig('vector_field.pdf')

def stem_with_color(x, y, ax, color, pos=True):
    
    if pos:
        marker = '^'
        x_ = x[y >= 0]
        y_ = y[y >= 0]
    else:
        marker = 'v'
        x_ = x[y < 0]
        y_ = y[y < 0]
        
    (markerline, stemlines, baseline) = plt.stem(x_, y_, markerfmt=marker)
    plt.setp(baseline, visible=False)
    plt.setp(markerline, color=color)
    plt.setp(stemlines, color=color)
    
def plot_interferences(N=2000):
    
    L = 200
    X, Y = np.meshgrid(np.linspace(0, L, N), np.linspace(-L/2, L/2, N))
    
    x1, y1 = 0, L/4
    x2, y2 = 0, -L/4
    patterns = np.sin(np.sqrt((X-x1)**2 + (Y-y1)**2)) + np.sin(np.sqrt((X-x2)**2 + (Y-y2)**2))
    
    patterns = np.c_[np.sin(X[:,:N//4])[::-1, :], patterns]
    
    plt.figure(figsize=(10, 5))
    plt.imshow(patterns, cmap=plt.cm.Blues_r)
    plt.axis('off')
    plt.savefig('patterns.pdf')
    
    plt.figure(figsize=(5, 5))
    plt.imshow(patterns[:, -1].reshape(-1,1), cmap=plt.cm.Blues_r, aspect=1/200)
    plt.axis('off')
    plt.savefig('screen.pdf')
    
    
def plot_filter(N=1000, Z=5E-6):
    c = 299792458/1.5
    lambdas, omegas = generate_wavelengths(N=N, c=c)

    #plot filter
    plt.figure(figsize=(3.45*1.4, 1.7*1.4))
#    plt.figure()
    fda.plot_h( lambdas, fda.s_z_tilde(Z, 2*np.pi*c/550E-9 -omegas), ax=plt.gca() )
    plt.savefig(fig_path + 's_z_tilde.pdf')   
    

     
    
if __name__ == '__main__':
    
    plot_pipeline()
#    plot_sinewaves(N=60)
#    plot_interferences()
#    plot_filter()
    
    
