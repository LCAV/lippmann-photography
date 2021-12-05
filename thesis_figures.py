# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 14:01:34 2017

@author: gbaechle
"""


import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
from lippmann import *
from finite_depth_analysis_bak import compare_with_filter, plot_h, s_z_tilde

import seaborn as sns
import seabornstyle as snsty

# snsty.setStyleMinorProject()

plt.close("all")


if __name__ == '__main__':
    Z = 5E-6
    N = 20000
    r = -1
    fig_path = "Figures/"

    plot_gaussian_lippmann_and_inverses()
    plt.figure()
    depths = np.linspace(0, Z*(1-1/N), N)

    lambdas, omegas = generate_wavelengths(N)

    spectrum = generate_gaussian_spectrum(lambdas=lambdas, mu=550E-9, sigma=30E-9)
    spectrum /= np.max(spectrum)

    lippmann, delta_lippmann = lippmann_transform(lambdas, spectrum, depths, r=r)
    show_lippmann_transform(depths, lippmann, ax=plt.gca())
    plt.title('Lippmann transform')
    plt.show()

    spectrum_shape = 'gauss'
    #    lambdas*=1E9

    #    h_filt = plot_h( lambdas, np.abs(h(lambdas, lambda_prime=550E-9, Z=Z))**2 )
    #    h_filt = plot_h( lambdas, h(lambdas, lambda_prime=550E-9, Z=Z) )
    h_filt = plot_h(lambdas, s_z_tilde(Z, 2 * np.pi * c / 550E-9 - omegas))
    # h_filt = plot_h(lambdas, s_z_tilde(50E-6, 2 * np.pi * c / 550E-9 - omegas))
    # h_filt = plot_h(lambdas, s_z_tilde(500E-6, 2 * np.pi * c / 550E-9 - omegas))
    # compare_with_filter(Z, N=N)




