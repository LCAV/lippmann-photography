# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 20:55:47 2017

@author: gbaechle
"""

import numpy as np
from lippmann import *
from multilayer_optics_matrix_theory import *
import matplotlib.pyplot as plt

sys.path.append("../")
from color_tools import *

fig_path = 'Figures/'



def semi_transparent_mirror_matrix(r):
    t = np.sqrt(1-r**2)
    return np.array([[1, 1j*r], [-1j*r, 1]])/t


def fabry_perot_display(wavelengths, d, r1, r2=None, plot=True):
    
    n0 = 1    
    if r2 is None:
        r2 = r1

    reflected_spectrum = []
    
    for wavelength in wavelengths:
                
        M1 = semi_transparent_mirror_matrix(r1)
        M2 = semi_transparent_mirror_matrix(r2)
        
        k = 2*np.pi/wavelength
        phi = n0*k*d
        P = np.array([[np.exp(-1j * phi), 0], [0, np.exp(1j * phi)]])  
        
        M = M2 @ P @ M1
        S = from_S_to_M(M)
    
        r = S[0, 1]
            
        reflected_spectrum.append(np.abs(r)**2)
        
    if plot:
        plt.figure()
        show_spectrum(wavelengths, reflected_spectrum, ax=plt.gca())
        plt.savefig('fabry_perot.pdf')
        plt.show()
    
    return reflected_spectrum
    
    
def fabry_perot_colors(ds, rs, N=300, plot=True, transmittance=False):
    
    colors_xyz = np.zeros((len(ds), len(rs), 3))
    wavelengths, _ = generate_wavelengths(N)   
    
    for i, d in enumerate(ds):
        for j, r in enumerate(rs):
            print(i,j)
            reflected_spectrum = np.array(fabry_perot_display(wavelengths, d, r, plot=False))
            if transmittance:
                reflected_spectrum = np.sqrt(1-reflected_spectrum**2)
            colors_xyz[i,j,:] = from_spectrum_to_xyz(wavelengths, reflected_spectrum)
           
    colors = from_xyz_to_rgb(colors_xyz)
    if plot:
        plt.figure()
        plt.imshow(colors, interpolation='none', extent=[rs[0],rs[-1],ds[-1]*1E6,ds[0]**1E6], aspect="auto")
        if transmittance:  
            plt.title('Transmitted spectrum')
            plt.xlabel('mirrors reflectance')
            plt.ylabel('mirrors distance ($\mu m$)')
            plt.show()
            plt.savefig(fig_path + 'fabry_perot_colors_t.pdf')
        else:
            plt.title('Reflected spectrum')
            plt.xlabel('mirrors reflectance')
            plt.ylabel('mirrors distance ($\mu m$)')
            plt.show()
            plt.savefig(fig_path + 'fabry_perot_colors_r.pdf')
            
    return colors
    

if __name__ == '__main__':

    lambdas, omegas = generate_wavelengths(N=300)
    
    fabry_perot_display(lambdas, d=300E-9, r1=0.1)
    
    N_d = 100
    N_r = 100
    ds = np.linspace(50E-9, 1000E-9, N_d)
    rs = np.linspace(0.05, 0.95, N_r)
    
    colors_r = fabry_perot_colors(ds, rs)
#    colors_t = fabry_perot_colors(ds, rs, transmittance=True)
    
    
    