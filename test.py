# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import seaborn as sns

from lippmann import *
from tools import *

sns.set_palette("Blues_r")
sns.set_style("whitegrid")

sns.set_context("notebook", font_scale=1.0, rc={"lines.linewidth": 3.0})


#close all figures
plt.close("all")
gui_manager = None


path = '/Users/gbaechle/EPFL/PhD/BRDF Data and Rendering Engines/CAVE - multispectral image database/fake_and_real_strawberries_ms/fake_and_real_strawberries_ms'
n = 1.0
integration_method='sum'

n_freqs = 34
n_space = 3000
n_time  = 200

c       = 299792458                             #Speed of light
lambdas = np.linspace(390E-9, 700E-9, n_freqs)  #Wavelengths

nus     = np.linspace(10E12, 770E12, np.int(n_freqs/34.*77))
#nus     = np.linspace(430E12, 770E12, n_freqs)
lambdas = np.sort(c/nus)

k       = 2*np.pi/lambdas                       #angular wavenumber
omega   = k*c                                   #angular frequency

n_freqs = len(lambdas)

intensities = np.zeros(n_freqs)

lippmann_temp = LippmannPlateDiscrete(n_freqs, 2, 2)

#lambdas = lippmann_temp.lambdas
#z       = lippmann_temp.z
#n_space = len(z)

#n_space = 1.0/(dz*dnu)

# 'Gaussian' spectrum
mu      = 573E-9
std_dev =  30E-9
#intensities = 1./(np.sqrt(2.*np.pi)*std_dev)*np.exp(-np.power((lambdas - mu)/std_dev, 2.)/2.) + \
#              1.5/(np.sqrt(2.*np.pi)*20E-9)*np.exp(-np.power((lambdas - 435E-9)/20E-9, 2.)/2.)

# MONOCHROMATIC LIGHT
#intensities[550-390] += 1

# BICHROMATIC LIGHT
idx = ((np.array([550, 470])-390)/310.*n_freqs).astype(np.int)
intensities[idx] += 1

#Normalize intensitiesd
intensities = intensities/np.max(intensities)

lippmann = LippmannPixel(Spectrum(lambdas, intensities), n)

lippmann_discrete = LippmannPixelDiscrete(intensities, reverse=True)

# LOAD SPECTRUM FROM CAVE DATABASE
lippmann_plate = load_multispectral_image_CAVE(path)
#strawberry
#lippmann = lippmann_plate.getpixel(420, 440)
#'tige'
#lippmann = lippmann_plate.getpixel(290, 410)
#blue
#lippmann = lippmann_plate.getpixel(10, 10)

#lippmann.object_spectrum.intensities = lippmann.object_spectrum.intensities/np.max(lippmann.object_spectrum.intensities)



#n_space = 2*np.floor(n_freqs*770/340)
#z = np.linspace(0, n_space/np.max(4./lambdas), n_space)
t = np.linspace(-1.0,1.0,n_time)*2*np.mean(lambdas)/lippmann.C
z = np.linspace(0, 300.0E-6, n_space)

Z = np.max(z)

x          = np.zeros((n_space, 3))
x[:,2]     = z
k_vec      = np.zeros((n_freqs, 3))
k_vec[:,2] = k

#E = lippmann.get_electric_field(t, z, display=True)
I = lippmann.get_intensity(z, integration_method=integration_method)
R = lippmann.get_reflectivity(z, integration_method=integration_method)


I_d = lippmann_discrete.get_intensity()
R_d = lippmann_discrete.get_reflectivity()

#standing_wave = lippmann3d.get_standing_wave(t, x, k_vec, display=True)
#em_field      = lippmann3d.get_electromagnetic_field(t, x, k_vec, display=True)

#fig = plt.figure()
#plt.plot(z,E, label='E')
#plt.plot(z,I, label='I')
#plt.legend()

lippmann.object_spectrum.show(title='Spectrum')

# Show Electric field over t and z
#fig = plt.figure()
#plt.imshow(E.T)
#plt.xlabel('z')
#plt.ylabel('t')

# Show Electric field for fixed t
#Spectrogram(z, E[:,10]).show(title='Electric field (for a fixed t)')

#Spectrogram(z, standing_wave.I).show(title='Intensity (new)')



#I.show(title='Intensity')
R.show(title='Reflectivity function')
R_d.show(title='Reflectivity function (discrete)')

# RE-ILLUMINATE
E_new, I_new = lippmann.re_illuminate(0.1, integration_method=integration_method)

I_d_new = lippmann_discrete.re_illuminate()

#normalize
#I_new.intensities = (I_new.intensities/np.max(I_new.intensities))

I_new.show(title='Recovered spectrum', sqrt=True)

I_d_new.show(title='Recovered spectrum (discrete)', sqrt=True)

max_val1 = np.max(np.sqrt(I_new.intensities))
max_val2 = np.max(lippmann.object_spectrum.intensities)
print 'Mean squared error:' + str(np.sum( (np.sqrt(I_new.intensities)/max_val1 - lippmann.object_spectrum.intensities/max_val2)**2 )/len(I_new.intensities) )




