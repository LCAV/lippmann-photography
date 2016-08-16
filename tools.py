# -*- coding: utf-8 -*-
import glob
import numpy as np
from scipy import misc, io
import matplotlib.pyplot as plt
from skimage.transform import PiecewiseAffineTransform, warp

from lippmann import *
import color_tools as ct

def load_multispectral_image_CAVE(path, image_type='png'):
    
    list_images = glob.glob(path + '/*.' + image_type)
    
    image = misc.imread(list_images[0]).astype(float)/255.0
    
    n_bands = 31
    
    wavelengths = np.linspace(400E-9, 700E-9, n_bands)
    
    lippmann_plate = LippmannPlate(wavelengths, image.shape[0], image.shape[1])
    lippmann_plate.rgb_ref = misc.imread(glob.glob(path + '/*.bmp')[0]).astype(float)/255.0
    
    idx = 0
    for idx, image_name in enumerate(list_images):
        
        image = misc.imread(image_name).astype(float)/255.0

        #gamma 'uncorrection'
#        image = np.power(image, 2.2)

        lippmann_plate.spectrums[idx] = image
        
        
    return lippmann_plate
    
    
def load_multispectral_image_SCIEN(path):

    mat_data    = io.loadmat(path)
    wavelengths = mat_data['wave'].flatten()*1E-9
    intensities = mat_data['photons']
        
    lippmann_plate = LippmannPlate(wavelengths, intensities.shape[0], intensities.shape[1])
    lippmann_plate.spectrums = Spectrums(wavelengths, intensities) 
    
    lippmann_plate.rgb_ref = lippmann_plate.spectrums.compute_rgb()
    
    return lippmann_plate
    
    
def load_multispectral_image_Suwannee(path):

    mat_data    = io.loadmat(path)
    wavelengths = mat_data['HDR']['wavelength'][0][0][0]*1E-9
    intensities = mat_data['I']
        
    lippmann_plate = LippmannPlate(wavelengths, intensities.shape[0], intensities.shape[1])
    lippmann_plate.spectrums = Spectrums(wavelengths, intensities) 
    
    lippmann_plate.rgb_ref = lippmann_plate.spectrums.compute_rgb()
    
    return lippmann_plate


def create_multispectral_image_discrete(path, N_prime):
    
    #read image
    im       = misc.imread(path).astype(float)/255.0
    
    #crate Lippmann plate object
    lippmann_plate = LippmannPlateDiscrete( N_prime, im.shape[0], im.shape[1])    
    
    #comppute the spectrum
    im_xyz   = ct.from_rgb_to_xyz(im)   
    lippmann_plate.spectrums.intensities = ct.from_xyz_to_spectrum(im_xyz, lippmann_plate.lambdas)
    
    return lippmann_plate
    
    
def create_multispectral_image(path, N_prime):
    
    #read image
    im       = misc.imread(path).astype(float)/255.0
    
    wavelengths = np.linspace(390-9, 700E-9, N_prime)
    
    #crate Lippmann plate object
    lippmann_plate = LippmannPlate( wavelengths, im.shape[0], im.shape[1])    
    
    #comppute the spectrum
    im_xyz   = ct.from_rgb_to_xyz(im)   
    lippmann_plate.spectrums.intensities = ct.from_xyz_to_spectrum(im_xyz, wavelengths)
    
    return lippmann_plate
    
    
def extract_layers_for_artwork(lippmann_plate, row_idx):
    
    plt.imsave('image_slice.png', lippmann_plate.spectrums.rgb_colors[:row_idx+1,:,:])
    
    r   = lippmann_plate.reflectances
    
    row = r[row_idx,:,:].T    
    #remove the mean
    row-= np.mean(row, axis=0)[np.newaxis, :]
    plt.imsave('front_slice.png', 1.-np.power(np.abs(row), 1./3.) )
    
    col = r[:row_idx+1,0,:].T
    #remove the mean
    col-= np.mean(col, axis=0)[np.newaxis, :]
    plt.imsave('left_slice.png', 1.-np.power(np.abs(col/np.max(row)), 1./3.), vmax=1., vmin=0.)


def image_perspective_transform(im, angle=np.pi/4, d=0.):
    
    nbre_samples = 10 
    
    rows = im.shape[0]
    cols = im.shape[1]

    if d == 0.:
        d   = im.shape[1]*10
        
    h       = im.shape[0]
    l       = im.shape[1]
    h1      = np.cos(angle)*h
    delta_d = np.sin(angle)*h
    h2 = d*h1/(d + delta_d)
    l2 = d*l/(d + delta_d)
    
#    l2 = h2
    
    src_row = np.linspace(0, h, nbre_samples)
    src_col = np.linspace(0, l, nbre_samples)
    
    src_row, src_col = np.meshgrid(src_row, src_col)
    src = np.dstack([src_col.flat, src_row.flat])[0]
    
    dst_row = h-np.linspace(h2, 0, nbre_samples)
    dst_col = np.linspace(0, l, nbre_samples)
    
    dst_row, dst_col = np.meshgrid(dst_row, dst_col)
    
    scale = np.linspace(l2/l, 1, nbre_samples)
    shift = np.linspace(l-l2, 0, nbre_samples)/2.

    dst_col = dst_col*scale[np.newaxis,:]+shift[np.newaxis,:]
    
    dst = np.dstack([dst_col.flat, dst_row.flat])[0]
    
    transform = PiecewiseAffineTransform()
    transform.estimate(dst, src)
    
    return warp(im, transform, output_shape=im.shape)

    
    
    
    

