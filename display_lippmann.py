# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from gui_manager import GuiManager

sns.set_palette("Blues_r")
sns.set_style("whitegrid")

sns.set_context("notebook", font_scale=1.0, rc={"lines.linewidth": 3.0})


#close all figures
plt.close("all")
gui_manager = None

N_samples = 100
theta_o   = np.pi/4.
theta_o   = 25./180.*np.pi
theta_o = 0.


path_CAVE = '/Users/gbaechle/EPFL/PhD/BRDF Data and Rendering Engines/CAVE - multispectral image database/fake_and_real_strawberries_ms/fake_and_real_strawberries_ms'
#path_CAVE = '/Users/gbaechle/EPFL/PhD/BRDF Data and Rendering Engines/CAVE - multispectral image database/fake_and_real_peppers_ms/fake_and_real_peppers_ms'
#path_CAVE = '/Users/gbaechle/EPFL/PhD/BRDF Data and Rendering Engines/CAVE - multispectral image database/egyptian_statue_ms/egyptian_statue_ms'
#path_CAVE = '/Users/gbaechle/EPFL/PhD/BRDF Data and Rendering Engines/CAVE - multispectral image database/pompoms_ms/pompoms_ms'
#path_CAVE = '/Users/gbaechle/EPFL/PhD/BRDF Data and Rendering Engines/CAVE - multispectral image database/fake_and_real_beers_ms/fake_and_real_beers_ms'

path_SCIEN = '/Users/gbaechle/EPFL/PhD/BRDF Data and Rendering Engines/SCIEN - Stanford multispectral image database/StanfordTower.mat'

path_Suwannee = '/Users/gbaechle/EPFL/PhD/BRDF Data and Rendering Engines/Multispectral image databases/Gulf_Wetlands_Sample_Rad/Suwannee_0609-1331_rad_small.mat'

path_PURDUE = '/Users/gbaechle/EPFL/PhD/BRDF Data and Rendering Engines/Purdue - DC'



path_RGB = 'images/final_small2.jpg'
path_RGB = 'images/final_raw_small.jpg'

# LOAD SPECTRUM FROM CAVE DATABASE
#try:
#    lippmann_plate
#except NameError:
from tools import *
from lippmann import *

lippmann_plate = load_multispectral_image_CAVE(path_CAVE)
#lippmann_plate = load_multispectral_image_SCIEN(path_SCIEN)
#lippmann_plate = load_multispectral_image_Suwannee(path_Suwannee)
#lippmann_plate = load_multispectral_image_PURDUE(path_PURDUE)

#Convert to discrete uniformly-sampled spectrums
lippmann_plate = lippmann_plate.to_uniform_freq(N_samples)

#Read an RGB image
#lippmann_plate = create_multispectral_image_discrete(path_RGB, N_samples)


#compute intensity and new spectrum
lippmann_plate.compute_new_spectrum()

#shift of the spectrum towards the blues
lippmann_plate.new_spectrums.blue_shift(1./np.cos(theta_o), extrapolation='cste')

lippmann_plate.spectrums.compute_rgb()
lippmann_plate.new_spectrums.compute_rgb(sqrt=True)

gui_manager = GuiManager(lippmann_plate, normalize_spectrums=True, gamma_correct=False)
gui_manager.show()


#extract_layers_for_artwork(lippmann_plate, 660)



