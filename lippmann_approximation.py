# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 13:54:11 2017

@author: gbaechle
"""

import numpy as np

from lippmann import *

import matplotlib.pyplot as plt
import seaborn as sns
import seabornstyle as snsty
snsty.setStyleMinorProject()

import sys
sys.path.append("../")
from multilayer_optics_matrix_theory import *

plt.rc('text', usetex=True)
plt.rc('font', family='serif')


if __name__ == '__main__':
    
    delta_z = 10E-9
    epsilon = 0.8E7
    Z = 5E-6
    n0 = 1.45

    lambdas, omegas = generate_wavelengths(N=500)
    depths = generate_depths(delta_z, Z)
    spectrum = generate_gaussian_spectrum(lambdas=lambdas, mu=550E-9, sigma=30E-9)

    intensity, delta_intensity = lippmann_transform(lambdas/n0, spectrum, depths)
    inverse_lippmann = inverse_lippmann(intensity, lambdas/n0, depths)
 
    r_approx, _ = propagation_arbitrary_layers_Lippmann_spectrum(rs=intensity, d=delta_z, lambdas=lambdas, plot=False, epsilon=epsilon, approximation=True)
    r, _ = propagation_arbitrary_layers_Lippmann_spectrum(rs=intensity, d=delta_z, lambdas=lambdas, plot=False, epsilon=epsilon, approximation=False)
 
    show_spectrum(lambdas, r_approx); plt.title('reflectance approx') 
    show_spectrum(lambdas, r); plt.title('reflectance')
    show_spectrum(lambdas, r-r_approx, vmax=1.1*np.max(r)); plt.title('difference')
    show_spectrum(lambdas, inverse_lippmann); plt.title('Lippmann inverse')
