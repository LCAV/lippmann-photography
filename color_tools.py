# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 14:25:04 2016

@author: gbaechle
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import nnls
from skimage.color import rgb2xyz, xyz2rgb
import matplotlib.pyplot as plt

from spectrum import *


def reconstruct_spectrum_from_rgb_shifts(rgb_images, angles, wavelengths):
    
    shape = rgb_images.shape
    measured = np.zeros([shape[0], shape[1], len(angles)*shape[2]])
    sensing_matrix   = np.zeros([len(angles)*shape[2], len(wavelengths)])
    
    xyz_images = np.zeros(shape)
    
    plt.figure()
    
    for i, angle in enumerate( angles ):
        
        xyz_images[:,:,:,i] = from_rgb_to_xyz(rgb_images[:,:,:,i])
        
        cmf_cie_x, cmf_cie_y, cmf_cie_z = shift_cmf_cie(wavelengths=wavelengths, factor=1./np.cos(angle))
        sensing_matrix[3*i+0, :]          = cmf_cie_x
        sensing_matrix[3*i+1, :]          = cmf_cie_y
        sensing_matrix[3*i+2, :]          = cmf_cie_z
        
        if i == 0:
            plt.plot(wavelengths, cmf_cie_x, lw=10.)
        else:
            plt.plot(wavelengths, cmf_cie_x)
        
        
        #extract each channel separately
        for j in range(shape[2]):
        
            measured[:,:,shape[2]*i+j] = xyz_images[:,:,j,i]

    measured  = np.reshape(measured, [-1, len(angles)*shape[2]]) 
    
    
    plt.figure()
    plt.imshow(sensing_matrix, interpolation='none'); plt.title('sensing matrix')
    plt.figure()
    plt.plot(measured.T); plt.title('measured signal')  
    plt.figure()
    plt.plot(np.reshape(rgb_images, [shape[0]*shape[1],3*len(angles)]).T); plt.title('RGB colors')  
    plt.figure()
    plt.plot(np.reshape(xyz_images, [shape[0]*shape[1],3*len(angles)]).T); plt.title('XYZ colors')
    
    #standard least squares
    spectrums = np.linalg.lstsq(sensing_matrix, measured.T)[0].T
    
    plt.figure()
    plt.plot(spectrums.T); plt.title('recovered spectrums (LS)')
    
    #non negative least squares
    spectrums = np.zeros( [shape[0]*shape[1], len(wavelengths)] )
    for row_idx in range(measured.shape[0]):
        row = measured[row_idx, :]
        
        spectrums[row_idx, :] = nnls(sensing_matrix, row)[0]
        
    plt.figure()
    plt.plot(spectrums.T); plt.title('recovered spectrums (NNLS)')
    
    
    print(sensing_matrix.shape)
    print(measured.shape)
    print(spectrums.shape)
    
    plt.figure()
    plt.plot(sensing_matrix.dot(spectrums.T)); plt.title('sensing matrix * spectrums')
    
    return Spectrums(wavelengths, np.reshape(spectrums, [shape[0], shape[1], len(wavelengths)]) )
    
    

def shift_cmf_cie(wavelengths, factor):
    
    shifted_wavelengths = wavelengths/factor    
    return read_cie_data(wavelengths=shifted_wavelengths)

def read_cie_data(wavelengths):
    
    cie_data = np.genfromtxt('cie_cmf.txt', delimiter='	')
    wavelengths_cie = cie_data[:,0]*1E-9     
    cmf_cie = cie_data[:,1:4]
        
    cmf_cie_interp_x = interp1d(wavelengths_cie, cmf_cie[:,0], kind='cubic', bounds_error=False, fill_value=0.0)(wavelengths)     
    cmf_cie_interp_y = interp1d(wavelengths_cie, cmf_cie[:,1], kind='cubic', bounds_error=False, fill_value=0.0)(wavelengths)
    cmf_cie_interp_z = interp1d(wavelengths_cie, cmf_cie[:,2], kind='cubic', bounds_error=False, fill_value=0.0)(wavelengths)      
  
    return cmf_cie_interp_x, cmf_cie_interp_y, cmf_cie_interp_z
    

def from_spectrum_to_xyz(wavelengths, spectral_colors):
      
    cmf_cie_x, cmf_cie_y, cmf_cie_z = read_cie_data(wavelengths=wavelengths) 
    
    X = np.trapz(y=spectral_colors*cmf_cie_x, x=wavelengths, axis=spectral_colors.ndim-1)
    Y = np.trapz(y=spectral_colors*cmf_cie_y, x=wavelengths, axis=spectral_colors.ndim-1)    
    Z = np.trapz(y=spectral_colors*cmf_cie_z, x=wavelengths, axis=spectral_colors.ndim-1)    
    
    #normalize
#    x = X/(X+Y+Z)
#    y = Y/(X+Y+Z)
#    z = Z/(X+Y+Z)
    
    #'normalize'
    normalization_cste = np.max(Y)
    X = X/normalization_cste
    Y = Y/normalization_cste
    Z = Z/normalization_cste
        
#    return np.stack([x,y,z], axis=-1)
    return np.stack([X,Y,Z], axis=-1)
    
def from_xyz_to_spectrum(xyz_colors, wavelengths):
    
    orig_shape = xyz_colors.shape
    xyz_colors = xyz_colors.reshape([-1,3])
    
    cie_data = np.genfromtxt('cie_cmf.txt', delimiter='	')
    wavelengths_cie = cie_data[:,0]*1E-9     
    cmf_cie = cie_data[:,1:4]
        
#    spectral_colors_interp = interp1d(wavelengths, spectral_colors, kind='cubic', bounds_error=False, fill_value=0.0)(wavelengths_cie)    
    cmf_cie_interp_x = interp1d(wavelengths_cie, cmf_cie[:,0], kind='cubic', bounds_error=False, fill_value=0.0)(wavelengths)     
    cmf_cie_interp_y = interp1d(wavelengths_cie, cmf_cie[:,1], kind='cubic', bounds_error=False, fill_value=0.0)(wavelengths)
    cmf_cie_interp_z = interp1d(wavelengths_cie, cmf_cie[:,2], kind='cubic', bounds_error=False, fill_value=0.0)(wavelengths)
    
    M = np.stack([cmf_cie_interp_x, cmf_cie_interp_y, cmf_cie_interp_z])
    
#    spectral_colors = np.zeros([xyz_colors.shape[0], len(wavelengths)])
    
    spectral_colors = np.linalg.lstsq(M, xyz_colors.T)[0].T
    
#    for idx in xrange(xyz_colors.shape[0]):
#        print idx
#        
##        spectral_colors[idx, :] = nnls(M, xyz_colors[idx,:])[0]
#        spectral_colors[idx, :] = np.linalg.lstsq(M, xyz_colors[idx,:])[0]
        
    return spectral_colors.reshape(orig_shape[:-1] + (len(wavelengths)))
    
    
def from_xyz_to_rgb(xyz_colors, normalize=True):
    
    rgb_colors = xyz2rgb(xyz_colors)
    
    if normalize:
        return rgb_colors/np.max(rgb_colors)
    else:
        return rgb_colors
    
def from_rgb_to_xyz(rgb_colors, normalize=True):
    
    xyz_colors = rgb2xyz(rgb_colors)    
    
    if normalize:
        return xyz_colors/np.max(xyz_colors)
    else:
        return xyz_colors
    
    

    