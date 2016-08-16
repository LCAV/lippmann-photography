# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 16:26:21 2016

@author: gbaechle
"""

import numpy as np
from scipy import misc, io
import matplotlib.pyplot as plt
import copy

from tools import *

plt.close("all")

theta = np.pi/2.

path_CAVE = '/Users/gbaechle/EPFL/PhD/BRDF Data and Rendering Engines/CAVE - multispectral image database/fake_and_real_strawberries_ms/fake_and_real_strawberries_ms'
path_Suwannee = '/Users/gbaechle/EPFL/PhD/BRDF Data and Rendering Engines/Multispectral image databases/Gulf_Wetlands_Sample_Rad/Suwannee_0609-1331_rad_small.mat'

path  = 'images/final_small.jpg'
save_images_path = 'images_video4/'
n_frames  = 100
n_samples = 34

lippmann_plate = create_multispectral_image_discrete(path, n_samples)
#lippmann_plate = load_multispectral_image_CAVE(path_CAVE)

#lippmann_plate = load_multispectral_image_Suwannee(path_Suwannee)

#lippmann_plate = lippmann_plate.to_uniform_freq(n_samples)
#lippmann_plate.compute_new_spectrum()

#im = misc.imread(path).astype(float)/255.0

angles = np.linspace(0, theta, n_frames)

shape = lippmann_plate.spectrums.intensities.shape
images = np.zeros((shape[0], shape[1], 3, n_frames))

for idx, angle in enumerate( angles ):
    
    print idx
    
    #create a copy of the plate
    lippmann_copy = copy.deepcopy(lippmann_plate)
    
    #shift of the spectrum towards the blues
    #lippmann_copy.spectrums.blue_shift(angle)
#    lippmann_copy.spectrums.blue_shift(1./np.cos(angle))
    lippmann_copy.spectrums.blue_shift(0.5 + 0.5/np.cos(angle))

    lippmann_copy.spectrums.rgb_colors = None
    lippmann_copy.spectrums.xyz_colors = None
    im2 = lippmann_copy.spectrums.compute_rgb(sqrt=True)
    
    #gamma correction
#    im2 = im2**2.2
    
#    im2 = image_perspective_transform(im2, angle=angle)
    images[:,:,:,idx] = im2*np.cos(angle)
    plt.imsave(save_images_path + '%.3d' %idx + '.png', im2)


image_diffuse = np.mean(images, axis=3)
plt.figure()
plt.imshow(image_diffuse)





