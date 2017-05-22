# -*- coding: utf-8 -*-
"""
Created on Thu May 18 19:12:57 2017

@author: gbaechle
"""

import numpy as np
import matplotlib.pyplot as plt

plt.close('all')


#can be used interchangably from S to M or from M to S
def from_S_to_M(S):
    
    return 1/S[1,1]*np.array([[S[0,0]*S[1,1]-S[0,1]*S[1,0], S[0,1]], [-S[1,0], 1]])

    
def Fresnel_equations(n1, n2, theta1, polarized='s'):
    
    cos_theta1 = np.cos(theta1)
    cos_theta2 = np.sqrt(1 -(n1/n2)**2 * np.sin(theta1)**2)
    
    if polarized == 's' or polarized == 'TE':
        r = (n1*cos_theta1 - n2*cos_theta2)/(n1*cos_theta1 + n2*cos_theta2)
        t = 1+r
        return r, t
        
    else:
        sec_theta1 = 1/cos_theta1
        sec_theta2 = 1/cos_theta2
        r = (n1*sec_theta1 - n2*sec_theta2)/(n1*sec_theta1 + n2*sec_theta2)
        t = (1+r)*cos_theta1/cos_theta2
        return r, t
        
def propagation_followed_by_boundary(n1, n2, phi):
    
     one_over_t = (n2+n1)*np.exp(1j*phi)
     r_over_t = (n2-n1)*np.exp(1j*phi)
     
     M = 1/(2*n2)*np.array([[np.conj(one_over_t), r_over_t], [np.conj(r_over_t), one_over_t]], dtype=complex)
     return M
    
        
def dielectric_Brag_grating(N, n1, n2, phi1, phi2):

    M1 = propagation_followed_by_boundary(n1, n2, phi1)
    M2 = propagation_followed_by_boundary(n2, n1, phi2)
    
    M0 = M2 @ M1
    
    M = np.linalg.matrix_power(M0, N)
    
    S = from_S_to_M(M)
    
    t = S[0,0]
    r = S[0,1]
    
    return r, t
    
def dielectric_Brag_grating_spectrum(N, n1, n2, d1, d2, resolution=1000, plot=True):     
    
    lambdas = np.linspace(390E-9, 700E-9, resolution)
    k = 2*np.pi/lambdas
    
    phis1 = n1*k*d1
    phis2 = n2*k*d2
    
    total_reflectance = []
    total_transmittance = []
    
    for (phi1, phi2) in zip(phis1, phis2):
        r, t = dielectric_Brag_grating(N, n1, n2, phi1, phi2)
        total_reflectance.append(np.abs(r)**2)
        total_transmittance.append(np.abs(t)**2)
        
    plt.figure()
    plt.plot(lambdas, total_reflectance)
    plt.show()
        
    return lambdas, total_reflectance, total_transmittance
    
    
def dielectric_Brag_grating_spectrum_unmatched_medium(d, N):
    
    pass
      
if __name__ == '__main__':
    
    N = 10
    n1 = 1.45
    n2 = 1.451
    n2 = 2
    d1 = 250E-9
    d2 = 250E-9
    
    dielectric_Brag_grating_spectrum(N, n1, n2, d1, d2)

    