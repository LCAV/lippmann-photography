# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 16:26:21 2016

@author: gbaechle
"""

import numpy as np
import seaborn as sns
from scipy import misc, io
import matplotlib.pyplot as plt
import copy
import color_tools as ct

sns.set_palette("Blues_r")
sns.set_style("whitegrid")

sns.set_context("notebook", font_scale=1.0, rc={"lines.linewidth": 3.0})

from tools import *
from gui_manager import GuiManager

plt.close("all")


path_CAVE = '/Users/gbaechle/EPFL/PhD/BRDF Data and Rendering Engines/CAVE - multispectral image database/fake_and_real_strawberries_ms/fake_and_real_strawberries_ms'
path_Suwannee = '/Users/gbaechle/EPFL/PhD/BRDF Data and Rendering Engines/Multispectral image databases/Gulf_Wetlands_Sample_Rad/Suwannee_0609-1331_rad_small.mat'
path_PURDUE = '/Users/gbaechle/EPFL/PhD/BRDF Data and Rendering Engines/Purdue - DC'

path  = 'images/final_small.jpg'
path  = 'images/final_raw_small.jpg'
save_images_path = 'frames_DC/'
n_frames  = 150
n_samples = 34

DIFFUSE = False
alpha   = 10     #angle of the prism (in DEGREES)
n1      = 1.45   #refraction index of glass
n2      = 1.0002 #refraction index of air

def from_viewing_angle_to_theta_i(theta_2, alpha, n1, n2, deg=True):
    
    #convert to radians
    if deg:
        theta_2 = np.deg2rad(theta_2)
        alpha   = np.deg2rad(alpha)
    
    theta_1_prime = theta_2+alpha
    theta_1       = np.arcsin(n2/n1*np.sin(theta_1_prime))
    theta_0_prime = alpha-theta_1
    
    if deg:
        return np.rad2deg(theta_0_prime)
    else:
        return theta_0_prime
   
#lippmann_plate = create_multispectral_image_discrete(path, n_samples)

#lippmann_plate = load_multispectral_image_CAVE(path_CAVE)
#lippmann_plate = load_multispectral_image_Suwannee(path_Suwannee)
lippmann_plate = load_multispectral_image_PURDUE(path_PURDUE)

lippmann_plate = lippmann_plate.to_uniform_freq(n_samples)

lippmann_plate.compute_new_spectrum()

#im = misc.imread(path).astype(float)/255.0

#angles = np.linspace(0, theta, n_frames)


r = 10   
z_max = 10-7.07
z_max = r
z = r - np.arange(n_frames)/(n_frames-1)*z_max
xplusy = np.sqrt( (r**2 - z**2) )
angles = np.pi/2.-np.arctan(z/xplusy)

theta_i = from_viewing_angle_to_theta_i(-angles, np.deg2rad(alpha), n1, n2, deg=False)

shape = lippmann_plate.spectrums.intensities.shape
images = np.zeros((shape[0], shape[1], 3, n_frames))



for idx, angle in enumerate( theta_i ):
    
    print(idx)
    
    #create a copy of the plate
    lippmann_copy = copy.deepcopy(lippmann_plate)
    
    #shift of the spectrum towards the blues
    #lippmann_copy.spectrums.blue_shift(angle)
    if not DIFFUSE:
        lippmann_copy.spectrums.blue_shift(1./np.cos(angle) )
    else:
        lippmann_copy.spectrums.blue_shift(0.5 + 0.5/np.cos(angle) )

    lippmann_copy.spectrums.rgb_colors = None
    lippmann_copy.spectrums.xyz_colors = None
    im2 = lippmann_copy.spectrums.compute_rgb(sqrt=False)

#    im2 = image_perspective_transform(im2, angle=theta_i)
#    images[:,:,:,idx] = im2*np.cos(angle)
    images[:,:,:,idx] = im2
    
    #gamma correction
#    im2 = im2**2.2

    plt.imsave(save_images_path + '%.3d' %idx + '.png', im2)


image_diffuse = np.mean(images, axis=3)
plt.figure()
plt.imshow(image_diffuse)

lippmann_plate.spectrums.compute_rgb()

wavelengths = lippmann_plate.spectrums.wave_lengths
new_spectrums = ct.reconstruct_spectrum_from_rgb_shifts(images, theta_i, wavelengths)
lippmann_plate.new_spectrums = new_spectrums

lippmann_plate.new_spectrums.compute_rgb(sqrt=False)

#show both spectrums
gui_manager = GuiManager(lippmann_plate, normalize_spectrums=False, gamma_correct=False)
gui_manager.show()


if DIFFUSE:
    plt.imsave(save_images_path + 'diffuse.png', image_diffuse**2.2)




