# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import seaborn as sns

import lippmann as lip1
import lippmann2 as lip2
from tools import *

sns.set_palette("Blues_r")
sns.set_style("whitegrid")

sns.set_context("notebook", font_scale=1.0, rc={"lines.linewidth": 3.0})


#close all figures
plt.close("all")
gui_manager = None

path_CAVE = '/Users/gbaechle/EPFL/PhD/BRDF Data and Rendering Engines/CAVE - multispectral image database/fake_and_real_strawberries_ms/fake_and_real_strawberries_ms'

lippmann_plate = load_multispectral_image_CAVE(path_CAVE)
spectrum = lippmann_plate.spectrums

lippmann = lip2.LippmannContinuous(spectrum.wave_lengths, 10, 10)

print(lippmann.phases().shape)
print(lippmann.complex_amplitude().shape)

print(lippmann.compute_wave_function(np.linspace(0, 100E-9, 100)).shape)

print(lippmann.I.shape)
print(lippmann.replay())

lippmann_discrete = lippmann.to_uniform_freq(34)
