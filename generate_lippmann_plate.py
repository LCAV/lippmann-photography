# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 16:27:58 2017

@author: gbaechle
"""

from scipy import misc, io
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from skimage.color import rgb2xyz, xyz2rgb
from lippmann import *

import sys
sys.path.append("../")
import color_tools as ct

plt.close('all')



def read_image(path):

    return misc.imread(path).astype(float)/255.
    
    
def compute_spectrum_slice(sliced, lambdas):
    
    #comppute the spectrum
    im_xyz   = xyz2rgb(sliced.reshape((1,-1,3))).reshape(-1, 3)
    spectrum = ct.from_xyz_to_spectrum(im_xyz, lambdas)
    
    return spectrum
    

def compute_lippmann_slice(spectrums, lambdas, depths):
    
    lippmann = np.zeros((len(spectrums), len(depths)))
    
    for i, s in enumerate(spectrums):
        print(i)
        lip, _ = lippmann_transform(lambdas, s, depths) 
        lippmann[i, :] = lip
        
    return lippmann
    
    
def compute_end_plate(im, lambdas, vmax):
    
    two_k = 4 * np.pi / lambdas
    
    im_xyz   = xyz2rgb(im)
    spectrums = ct.from_xyz_to_spectrum(im_xyz, lambdas)
    
    intensity = -np.trapz(spectrums, two_k*c/2, axis=2)
    mpl.image.imsave('Figures/baseline.png', intensity, vmax=vmax, vmin=0)
    
    return intensity

    
def generate_slices(im, N=500):
    
    lambdas, _ = generate_wavelengths(N)
    depths = generate_depths(delta_z=2.5E-9, max_depth=2.5E-6)
    
    H = 883-1
    L = 883-1
    slice1 = compute_spectrum_slice(im[:H, L, :3], lambdas)
    slice2 = compute_spectrum_slice(im[H, :L, :3], lambdas)
    slice3 = compute_spectrum_slice(im[:H, 0, :3], lambdas)
    slice4 = compute_spectrum_slice(im[0, :L, :3], lambdas)
    
    lip1 = compute_lippmann_slice(slice1, lambdas, depths)
    lip2 = compute_lippmann_slice(slice2, lambdas, depths)
    lip3 = compute_lippmann_slice(slice3, lambdas, depths)
    lip4 = compute_lippmann_slice(slice4, lambdas, depths)
    
    print(np.max(lip1), np.max(lip2), np.max(lip3), np.max(lip4))
    vmax = max(np.max(lip1), np.max(lip2), np.max(lip3), np.max(lip4))
    
    for i in range(1,5):    
        
        i_str = str(i)
        mpl.image.imsave('Figures/slice' + i_str + '.png', eval('lip' + i_str), vmax=vmax)
    
    return lambdas, vmax
    
    
    
if __name__ == '__main__':
    
#    path = '../images/original.png'
    path = '../images/lippmann_image.jpg'
    im = read_image(path)   
    
    lambdas, vmax = generate_slices(im, N=500)
    
#    spectrum = compute_end_plate(im[:800, :750, :3], lambdas, vmax) 
    spectrum = compute_end_plate(im[:, :, :3], lambdas, vmax) 
    
#    misc.imsave('Figures/front.png', im[:800, :750])
    misc.imsave('Figures/front.png', im)
    
    plt.figure()
    plt.imshow(im)
    plt.figure()
#    plt.imshow(im[:800, :750, :3])
    plt.imshow(im[:, :, :3])
    
    