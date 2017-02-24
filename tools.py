# -*- coding: utf-8 -*-
import glob
import numpy as np
from scipy import misc, io
import matplotlib.pyplot as plt
from skimage.transform import PiecewiseAffineTransform, warp
from osgeo import gdal
import csv
import h5py

from lippmann import *
import color_tools as ct


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


def load_multispectral_image_PURDUE(path):
    
    gtif = gdal.Open( path + "/data.tif" )
    
    #extract wavelengths
    wavelength_data = np.genfromtxt( path + "/wavelengths.txt", delimiter=' ')
    indices = np.where( 1-np.isnan(wavelength_data[:,2]) )
    wavelengths = wavelength_data[indices, 1].flatten()*1E-9
        
    shape = gtif.GetRasterBand(1).GetDataset().ReadAsArray()[0].shape    
    lippmann_plate = LippmannPlate(wavelengths, shape[1], shape[0]//2)
#    lippmann_plate = LippmannPlate(wavelengths, 1, 1)
    
    for idx in range( gtif.RasterCount ):
        print("[ GETTING BAND ]: ", idx)
        band = gtif.GetRasterBand(idx+1)
        
        data = band.GetDataset().ReadAsArray()[idx]

        #reduce the shape        
        lippmann_plate.spectrums[idx] = data[shape[0]//2:, :].transpose()


    return lippmann_plate
    

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
    
def load_multispectral_image_Gamaya(path, filename):
    
    with open(path + '/wavs.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        wavelengths = np.array(next(reader), dtype=float)*1E-9
        
    gtif = gdal.Open(path + '/' + filename)
    shape = gtif.GetRasterBand(1).GetDataset().ReadAsArray()[0].shape    
    lippmann_plate = LippmannPlate(wavelengths, shape[0], shape[1])
    
    for idx in range( gtif.RasterCount ):
        band = gtif.GetRasterBand(idx+1)
        
        data = np.float_(band.GetDataset().ReadAsArray()[idx])/255.
        data = np.nan_to_num(data)
        data[data<0] = 0.
        
        lippmann_plate.spectrums[idx] = data
    
    return lippmann_plate
    

def load_multispectral_image_HySpex(path):
    
    f = h5py.File(path)
 
    data = np.array(f['D'])  
    wavelengths = np.array(f['wavelengths']).flatten()*1E-9
    
    #put the spectral dimension at the end
    data = np.rollaxis(data, 2)
    data = np.rollaxis(data, 2)    

    print(data.shape)
    print(wavelengths)    
    
    #this dataset is huge, cut a bit... or not
    intensity = data[600:1200,350:950,:]
    
    #normalize
    intensity -= np.min(intensity)
    intensity /= np.max(intensity)
    
    lippmann_plate = LippmannPlate(wavelengths, intensity.shape[0], intensity.shape[1])
    lippmann_plate.spectrums = Spectrums(wavelengths, intensity) 
    
    lippmann_plate.rgb_ref = lippmann_plate.spectrums.compute_rgb()
    
    return lippmann_plate


def create_multispectral_image_discrete(path, N_prime):
    
    #read image
    im       = misc.imread(path).astype(float)/255.0
    
    #create Lippmann plate object
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
    
    
def extract_layers_for_artwork(lippmann_plate, row_idx, subtract_mean=True, normalize=False, negative=False):
    
    plt.imsave('image_slice.png', lippmann_plate.spectrums.rgb_colors[:row_idx+1,:,:])
    
    r   = lippmann_plate.reflectances
    min_r = np.min(r)
    max_r = np.max(r)
    
    row = r[row_idx,:,:].T    
    #remove the mean
    if subtract_mean:
        row-= np.mean(row, axis=0)[np.newaxis, :]
    if negative:
        min_r, max_r = -max_r, -min_r
        row = -row
    if normalize:
        plt.imsave('front_slice.png', row, vmin=min_r, vmax=max_r )
    else:    
        plt.imsave('front_slice.png', 1.-np.power(np.abs(row), 1./3.) )
    
    col = r[:row_idx+1,0,:].T
    #remove the mean
    if subtract_mean:
        col-= np.mean(col, axis=0)[np.newaxis, :]
    if negative:
        col = -col
    if normalize:
        plt.imsave('left_slice.png', col, vmin=min_r, vmax=max_r)
    else:
        plt.imsave('left_slice.png', 1.-np.power(np.abs(col/np.max(row)), 1./3.), vmax=1., vmin=0.)


def generate_interference_images(lippmann_plate):
    
    r   = lippmann_plate.reflectances
    min_r = np.min(r)
    max_r = np.max(r)
    
    for z in range(r.shape[2]):
        im = r[:,:,z]
        plt.imsave('interferences/'+str(z).zfill(3) + '.png', im, vmin=min_r, vmax=max_r)


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

    
    
    
    

