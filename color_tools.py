# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 14:25:04 2016

@author: gbaechle
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import nnls
from skimage.color import rgb2xyz, xyz2rgb

def from_spectrum_to_xyz(wavelengths, spectral_colors):

    cie_data = np.genfromtxt('cie_cmf.txt', delimiter='	')
    wavelengths_cie = cie_data[:,0]*1E-9     
    cmf_cie = cie_data[:,1:4]
        
#    spectral_colors_interp = interp1d(wavelengths, spectral_colors, kind='cubic', bounds_error=False, fill_value=0.0)(wavelengths_cie)    
    cmf_cie_interp_x = interp1d(wavelengths_cie, cmf_cie[:,0], kind='cubic', bounds_error=False, fill_value=0.0)(wavelengths)     
    cmf_cie_interp_y = interp1d(wavelengths_cie, cmf_cie[:,1], kind='cubic', bounds_error=False, fill_value=0.0)(wavelengths)
    cmf_cie_interp_z = interp1d(wavelengths_cie, cmf_cie[:,2], kind='cubic', bounds_error=False, fill_value=0.0)(wavelengths)      
    
    X = np.trapz(y=spectral_colors*cmf_cie_interp_x, x=wavelengths, axis=spectral_colors.ndim-1)
    Y = np.trapz(y=spectral_colors*cmf_cie_interp_y, x=wavelengths, axis=spectral_colors.ndim-1)    
    Z = np.trapz(y=spectral_colors*cmf_cie_interp_z, x=wavelengths, axis=spectral_colors.ndim-1)    
    
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
        
    return spectral_colors.reshape(orig_shape[:-1] + (len(wavelengths),))
    
    
def from_xyz_to_rgb(xyz_colors, normalize=True):
    
    rgb_colors = xyz2rgb(xyz_colors)
    
    if normalize:
        return rgb_colors/np.max(rgb_colors)
    else:
        return rgb_colors
    
def from_rgb_to_xyz(rgb_colors):
    
    return rgb2xyz(rgb_colors)
    
    

    