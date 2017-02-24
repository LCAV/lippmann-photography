# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 13:34:45 2016

@author: gbaechle
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.optimize import nnls
import scipy.integrate as integrate

from spectrum import *
from color_tools import *
from tools import *

plot_spectrums = False

plt.close("all")

n_channels = 3
n_frames   = 2
path = 'spectrum_white.p'

#load spectrum
with open(path, "rb") as f:
    spectrum = pickle.load(f)
    

def compute_inner_products(wavelengths, theta_k, max_theta_k, n, n_samples=1000):
    
    c = 299792458
    v_min = c/np.max(wavelengths)
    v_max = (c/np.min(wavelengths))/np.cos(max_theta_k)
    I     = v_max-v_min
    
    nu = np.linspace(v_min, v_max, n_samples)
    shifted_nu = nu/np.cos(theta_k)
    
    #get x, y, z functions
    #shifted_wavelengths = c/shifted_nu
    x, y, z = read_cie_data(wavelengths=c/nu)
    
    #compute cosine
    cosine = np.cos(shifted_nu*n*np.pi/I)
    
#    plt.figure()
#    plt.plot(x*cosine)
#    plt.plot(y*cosine)
#    plt.plot(z*cosine)
#    plt.plot(cosine, '--')
    
    x_ = integrate.trapz(y=x*cosine, x=shifted_nu)
    y_ = integrate.trapz(y=y*cosine, x=shifted_nu)
    z_ = integrate.trapz(y=z*cosine, x=shifted_nu)
    
#    x_ = np.dot(x, cosine)
#    y_ = np.dot(y, cosine)
#    z_ = np.dot(z, cosine)
    
    return x_, y_, z_
    

def build_cosine_series_matrix(angles, wavelengths, n0=20):

    sensing_matrix = np.zeros([len(angles)*n_channels, n0])
    
    for k, angle in enumerate(angles):
        print(angle)
        for n in range(n0):
            
            x,y,z = compute_inner_products(wavelengths, angle, angles[-1], n)
            sensing_matrix[k*3,n] = x
            sensing_matrix[k*3+1,n] = y
            sensing_matrix[k*3+2,n] = z
            
    return sensing_matrix
    

def cosine_series(wavelengths, spectrum, n0):
    
    c = 299792458    
    
    nu = c/wavelengths
    I  = np.max(nu)-np.min(nu)
    
    f_n = np.zeros(n0)
        
    for n in range(n0):
        
        cosine = np.cos(nu*n*np.pi/I)      
        f_n[n] = integrate.trapz(y=spectrum*cosine/I, x=nu)
        
    return f_n
    
    
def inverse_cosine_series(f_coeffs, wavelengths):
    
    c = 299792458
    
    nu = c/wavelengths
    I  = np.max(nu)-np.min(nu)
    
    spectrum = np.zeros(len(wavelengths))
    
    for n, f_n in enumerate(f_coeffs):
        if n == 0:
            spectrum += f_n*np.ones(len(wavelengths))
        else:
            spectrum += 2*f_n*np.cos(n*nu*np.pi/I)
            
    return spectrum
    
    
def build_discrete_matrix(angles):
    
    sensing_matrix = np.zeros([len(angles)*n_channels, len(wavelengths)])
    
    #build the matrix
    for i, angle in enumerate( angles ):
    
        cmf_cie_x, cmf_cie_y, cmf_cie_z = shift_cmf_cie(wavelengths=wavelengths, factor=1./np.cos(angle))
        sensing_matrix[3*i+0, :]        = cmf_cie_x
        sensing_matrix[3*i+1, :]        = cmf_cie_y
        sensing_matrix[3*i+2, :]        = cmf_cie_z
        
        if i == 0:
            plt.plot(wavelengths, cmf_cie_x, lw=10.)
        else:
            plt.plot(wavelengths, cmf_cie_x)
            
    return sensing_matrix
      

f_true = [3,2,1,0.3]

spec = inverse_cosine_series(f_true, spectrum.wave_lengths)
spectrum = Spectrum(spectrum.wave_lengths, spec)




#compute the CIE X,Y,Z values
wavelengths = spectrum.wave_lengths
cmf_cie_x, cmf_cie_y, cmf_cie_z = read_cie_data(wavelengths)


r = 10   
z_max = 10-7.07
z = r - np.arange(n_frames)/(n_frames-1)*z_max
xplusy = np.sqrt( (r**2 - z**2) )
angles = np.pi/2.-np.arctan(z/xplusy)

alpha = 10.
n1    = 1.45
n2    = 1.0002


theta_i = from_viewing_angle_to_theta_i(-angles, np.deg2rad(alpha), n1, n2, deg=False)

sensing_matrix   = build_discrete_matrix(theta_i)

plt.figure()
        
xyz_colors2 = np.dot(sensing_matrix, spectrum.intensities)

xyz_colors = np.zeros(n_channels*n_frames)
for k in range(n_frames):
    
    xyz = from_spectrum_to_xyz(wavelengths*np.cos(theta_i[k]), spectrum.intensities, normalize=False)
    
    print(xyz)
    
    xyz_colors[3*k+0] = xyz[0]
    xyz_colors[3*k+1] = xyz[1]
    xyz_colors[3*k+2] = xyz[2]

#xyz_colors = xyz_colors2

print('condition number of the sensing matrix: ', np.linalg.cond(sensing_matrix))


plt.figure()
plt.imshow(sensing_matrix, interpolation='none'); plt.title('sensing matrix')
plt.figure()
plt.imshow(np.reshape(xyz_colors, [-1, 3]), interpolation='none'); plt.title('measured XYZ colors')


if plot_spectrums:
    spectrum.show(title='original')

#standard least squares
spectrum_rec = Spectrum(wavelengths, np.linalg.lstsq(sensing_matrix, xyz_colors)[0] )

#NNLS with Tikhonov regularization
lambd = 10E-3

#L2 regularization
D = np.eye(len(wavelengths))

#Smoothness: difference operator
D = 2*np.eye(len(wavelengths)) - np.eye(len(wavelengths), k=1) - np.eye(len(wavelengths), k=-1)

A = np.concatenate((sensing_matrix, lambd*D) )
b = np.concatenate((xyz_colors, np.zeros(len(wavelengths)) ))
spectrum_rec = Spectrum(wavelengths, nnls(A, b)[0] )

print('condition number of the Tikhonov matrix: ', np.linalg.cond(A))

if plot_spectrums:
    spectrum_rec.show(title='recovered')

cos_matrix = build_cosine_series_matrix(theta_i, wavelengths, n0=8)
plt.figure()
plt.imshow(cos_matrix, interpolation='nearest')
plt.figure()
plt.plot(cos_matrix[0,:])
plt.plot(cos_matrix[-3,:]+0.1)
#print(cos_matrix)

print('condition number of the continuous sensing matrix: ', np.linalg.cond(cos_matrix))

f_coeffs = np.linalg.lstsq(cos_matrix, xyz_colors)[0]
#adjust f_0
#f_coeffs[0] = f_coeffs[0]*2

#inverse cosine series
spectrum_rec_cont = Spectrum(wavelengths, inverse_cosine_series(f_coeffs, wavelengths))

if plot_spectrums:
    spectrum_rec_cont.show(title='recovered continuous')

plt.figure()
plt.plot(xyz_colors2)
plt.title('discrete')

plt.figure()
plt.plot(xyz_colors)
plt.title('continuous')

#f_true = np.linspace(100,50,51)

wavelengths = np.linspace(390, 770, 100)*10E-9
wavelengths = 100/np.arange(100)
wavelengths = 1/np.linspace(1,50,51)

print(f_true)
spe = inverse_cosine_series(f_true, wavelengths)
f_n_rec = cosine_series(wavelengths, spe, len(f_true))
plt.figure()
plt.plot(wavelengths, spe)
print(f_n_rec)
plt.figure()
plt.plot(wavelengths, inverse_cosine_series(f_n_rec, wavelengths))



