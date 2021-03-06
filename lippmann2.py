# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 14:01:34 2017

@author: gbaechle
"""

from lippmann import *

import color_tools as ct
from scipy.special import erfc
import matplotlib.gridspec as gridspec


from cube import *
fig_path = 'Figures/'

def plot_fib(path, file, Z, n0=1.5):
    im = plt.imread(path + file)
    lippmann = np.mean(im, axis=1)
    depths = np.linspace(0, Z, len(lippmann))
    lambdas, omegas = generate_wavelengths(500)
            
    spectrum_replayed = inverse_lippmann(lippmann, lambdas/n0, depths)
#    col = ct.from_xyz_to_rgb( ct.from_spectrum_to_xyz(lambdas, spectrum_replayed).reshape((1,1,-1)) )
    
    plt.figure(figsize=(3.45, 2.5))
    show_lippmann_transform(depths, lippmann, ax=plt.gca()) 
    plt.title("Measured spectrum: " + file.replace('_', '\_')) 
#    plt.savefig(fig_path + 'LippmannFIB2.pdf') 
    
    plt.figure(figsize=(3.45, 2.5))
    show_spectrum(lambdas, spectrum_replayed, ax=plt.gca(), show_background=True) 
    plt.title("Inverse Lippmann: " + file.replace('_', '\_')) 
    
#    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(3.45/0.6, 3.45/2.5/0.6))
#    ax1.imshow(im.T, extent=(0, Z*1E6, 0, 1), aspect='auto', cmap='gray')
#    ax1.set_yticks([])
#    ax1.set_xlabel('Depth ($\mu m$)')
#    ax1.set_title('(a) FIB image')
#    show_lippmann_transform(depths, lippmann, ax=ax2) 
#    ax2.set_ylim([0.9*np.min(lippmann), 1.1*np.max(lippmann)])
#    ax2.set_title('(b) Column averaging')
#    show_spectrum(lambdas, spectrum_replayed, ax=ax3, show_background=True) 
#    ax3.set_title('(c) Synthesized spectrum')
#    plt.savefig(fig_path + 'LippmannFIB.pdf') 
    
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.45*1.6/1.1, 3.45/1.1))
    ax1.imshow(im.T, extent=(0, Z*1E6, 0, 1), aspect='auto', cmap='gray')
    ax1.set_yticks([])
    ax1.set_xlabel('Depth ($\mu m$)')
    ax1.set_title('(a) FIB image')
    show_lippmann_transform(depths, lippmann, ax=ax2) 
    ax2.set_ylim([0.9*np.min(lippmann), 1.1*np.max(lippmann)])
    ax2.set_title('(b) Column averaging')
    show_spectrum(lambdas, spectrum_replayed, ax=ax3, show_background=True) 
    plt.savefig(fig_path + 'LippmannFIB.pdf') 


def plot_fibs(n0=1.5):
    
    path = "/Users/gbaechle/EPFL/PhD/BRDF Data and Rendering Engines/CIME/"
    files = ["Lippmann2_11.jpg",
             "Lippmann_pen test06.jpg",
             "Lippmann3_brighter spot_07_2.jpg",
             "Lippmann3_green point_04_2.jpg",
             "Lippmann4_07.jpg"]
    Zs = [7.2E-6, 3.2E-6, 6.57E-6, 6.69E-6, 5.55E-6]
    
    for file, Z in zip(files, Zs):
        plot_fib(path, file, Z, n0)
        
    
    

def color_shift(n_wavelengths=100, ax=None):
    
    lambdas, omegas = generate_wavelengths(500) #3000
    depths = generate_depths(max_depth=7E-6)
        
    rs = [-1, 0.7*np.exp(1j*np.deg2rad(-148)), 0.2, 1]
    r_names = ['$r=-1$', 'Hg', 'Air', '$r=1$']
    
    rs = [0.7*np.exp(1j*np.deg2rad(148)), 0.2]
    r_names = ['Hg', 'Air']
    
    colors_XYZ = np.zeros((len(rs)+1, n_wavelengths, 3))
    
    modulation = np.linspace(1, 0.2, len(lambdas))
#    modulation = 0.5+np.random.rand(len(lambdas))
#    modulation = generate_gaussian_spectrum(lambdas, mu=450E-9, sigma=300E-9)
#    modulation = generate_gaussian_spectrum(lambdas, mu=450E-9, sigma=150E-9)
#    modulation = (np.exp(np.linspace(0,1,len(lambdas)))-1 )/(np.exp(1)-1)
    
    for i, lambd in enumerate( np.linspace(400E-9, 700E-9, n_wavelengths) ):
        print(i)
#        spectrum = generate_gaussian_spectrum(lambdas=lambdas, mu=lambd, sigma=30E-9)
        spectrum = generate_mono_spectrum(lambdas=lambdas, color=lambd)
        colors_XYZ[0, i, :] = -ct.from_spectrum_to_xyz(lambdas, spectrum, normalize=False)
        
#        spectrum *= modulation
        
        for j, r in enumerate(rs):
            window = erfc(np.linspace(0,3,len(depths))) 
            lippmann = lippmann_transform(lambdas, spectrum, depths, r=r)[0]*window
            spectrum_replayed = inverse_lippmann(lippmann, lambdas, depths)
            
            colors_XYZ[j+1, i, :] = -ct.from_spectrum_to_xyz(lambdas, spectrum_replayed, normalize=False)
        
    #normalize colors
    colors_xyz = colors_XYZ/np.max(np.sum(colors_XYZ, axis=2),axis=1)[:, None,None]
#    colors_xyz = colors_XYZ/np.max(colors_XYZ[:,:,1], axis=1)[:,None, None]
    colors_rgb = ct.from_xyz_to_rgb(colors_xyz, normalize=False)
    colors_rgb /= np.max(colors_rgb, axis=(1,2))[:,None,None]
    
#    colors_xyz = colors_XYZ/np.max(colors_XYZ[:,:,1])
#    colors_rgb = ct.from_xyz_to_rgb(colors_xyz, normalize=False)
#    colors_rgb /= np.max(colors_rgb)
        
    if ax is None:
        plt.figure(figsize=(3.45, 2.5))
        ax = plt.gca()
    
    ax.imshow(colors_rgb, extent=(400E-9, 700E-9, 0, 1), aspect='auto')
    L = len(rs)+1
    ax.set_yticks(np.linspace(1/(2*L), 1-1/(2*L), L))
    ax.set_yticklabels((['Original']+r_names)[::-1])
    ax.set_xticks(np.linspace(400E-9, 700E-9,4)); plt.gca().set_xticklabels(np.linspace(400,700,4).astype(int))
    ax.set_xlabel('Wavelength (nm)')
    
    plt.savefig(fig_path + 'color_shift_PNAS_erfc3_.pdf')


def color_shift_3_figs(n_wavelengths=100, N=10):
    
    rhos = color_shift_cube(wavelengths=np.linspace(400E-9, 700E-9, n_wavelengths), rhos=np.linspace(-1,1,N), thetas=np.array([0]), return_XYZ=True)
    thetas = color_shift_cube(wavelengths=np.linspace(400E-9, 700E-9, n_wavelengths), rhos=np.array([1]), thetas=np.linspace(0,np.pi,N), return_XYZ=True)
    
    rhos = rhos.reshape((N,n_wavelengths,3))
    thetas = thetas.reshape((N,n_wavelengths,3))
    
    max_rhos = np.max(rhos[:,:,1])
    rhos /= max_rhos
    thetas /= max_rhos
    
    rhos = ct.from_xyz_to_rgb(rhos, normalize=False)
    thetas = ct.from_xyz_to_rgb(thetas, normalize=False) 
    
    f, (ax1, ax2, ax3) = plt.subplots(3, figsize=(3.45, 1.5*3.45), sharex=True)
    
    ax1.imshow(rhos, extent=(400E-9, 700E-9, 0, 1), aspect='auto')
    ax1.set_yticks(np.linspace(0,1,3))
    ax1.set_yticklabels(np.linspace(1,-1,3))
    ax1.set_ylabel(r'$\rho$', rotation=0)
    ax1.set_title(r'(a) Color rendition with respect to $\rho$')
    
    ax2.imshow(thetas, extent=(400E-9, 700E-9, 0, 1), aspect='auto')
    ax2.set_yticks(np.linspace(0,1,3))
    ax2.set_yticklabels(['$\pi$','$\pi/2$', '$0$'])
    ax2.set_ylabel(r'$\theta$', rotation=0)
    ax2.set_title(r'(b) Color rendition with respect to $\theta$')
    
    color_shift(n_wavelengths, ax=ax3)
    ax3.set_title(r'(c) Color rendition for selected values of $r$')
    
    plt.savefig(fig_path + 'color_shift_3_figs.pdf')


def color_shift_2_figs(n_wavelengths=100, N=10):
    
    rhos = color_shift_cube(wavelengths=np.linspace(400E-9, 700E-9, n_wavelengths), rhos=np.linspace(-1,1,N), thetas=np.array([0]), return_XYZ=True)
    thetas = color_shift_cube(wavelengths=np.linspace(400E-9, 700E-9, n_wavelengths), rhos=np.array([1]), thetas=np.linspace(0,np.pi,N), return_XYZ=True)
    
    rhos = rhos.reshape((N,n_wavelengths,3))
    thetas = thetas.reshape((N,n_wavelengths,3))
    
    max_rhos = np.max(rhos[:,:,1])
    rhos /= max_rhos
    thetas /= max_rhos
    
    rhos = ct.from_xyz_to_rgb(rhos, normalize=False)
    thetas = ct.from_xyz_to_rgb(thetas, normalize=False) 
    
    f, (ax1, ax2) = plt.subplots(2, figsize=(1.3*3.45, 1.3*3.45), sharex=True)
    
    ax1.imshow(rhos, extent=(400E-9, 700E-9, 0, 1), aspect='auto')
    ax1.set_yticks(np.linspace(0,1,3))
    ax1.set_yticklabels(np.linspace(1,-1,3))
    ax1.set_ylabel(r'$\rho$', rotation=0)
    ax1.set_title(r'(a) Color rendition with respect to $\rho$')
    
    ax2.imshow(thetas, extent=(400E-9, 700E-9, 0, 1), aspect='auto')
    ax2.set_yticks(np.linspace(0,1,3))
    ax2.set_yticklabels(['$\pi$','$\pi/2$', '$0$'])
    ax2.set_ylabel(r'$\theta$', rotation=0)
    ax2.set_title(r'(b) Color rendition with respect to $\theta$')
    ax2.set_xticks(np.linspace(400E-9, 700E-9,4)); plt.gca().set_xticklabels(np.linspace(400,700,4).astype(int))
    ax2.set_xlabel('Wavelength (nm)')
    
    plt.savefig(fig_path + 'color_shift_2_figs.pdf')
    
    
    
def color_shift_cube(wavelengths=np.linspace(400E-9, 700E-9, 100), rhos = np.linspace(-1,1,10), thetas = np.linspace(0, np.pi, 10), return_XYZ=False):
    
    lambdas, omegas = generate_wavelengths(100) #3000
    depths = generate_depths(max_depth=5E-6)
    
    colors_XYZ = np.zeros((len(rhos), len(thetas), len(wavelengths), 3))
    window = erfc(np.linspace(0,4,len(depths))) 
    
    for n, lambd in enumerate(wavelengths):
        print(n)
        spectrum = generate_gaussian_spectrum(lambdas=lambdas, mu=lambd, sigma=30E-9)
#        spectrum = generate_mono_spectrum(lambdas=lambdas, color=lambd)
#        colors_XYZ[0, 0, n, :] = -ct.from_spectrum_to_xyz(lambdas, spectrum, normalize=False)
        
        for i, rho in enumerate(rhos):
            for j, theta in enumerate(thetas):
                
                r = rho*np.exp(-1j*theta)
                lippmann = lippmann_transform(lambdas, spectrum, depths, r=r)[0]#*window
                spectrum_replayed = inverse_lippmann(lippmann, lambdas, depths)    
                colors_XYZ[i,j,n,:] = -ct.from_spectrum_to_xyz(lambdas, spectrum_replayed, normalize=False)
        
    #normalize colors
#    colors_xyz = colors_XYZ/np.max(np.sum(colors_XYZ, axis=2),axis=1)[:, None,None]
    if return_XYZ:
        return colors_XYZ
    
    colors_xyz = colors_XYZ/np.max(colors_XYZ[:,:,:,1])
    colors_rgb = ct.from_xyz_to_rgb(colors_xyz, normalize=False)
    colors_rgb /= np.max(colors_rgb)
    
    if len(rhos) == 1 :
        plt.figure(figsize=(3.45, 2.5))
        plt.imshow(colors_rgb[0,:,:,:], extent=(400E-9, 700E-9, 0, 1), aspect=(300E-9)/2)
        plt.gca().set_yticks(np.linspace(0,1,3))
        plt.gca().set_yticklabels(['0','$\pi/2$', '\pi'])
        plt.gca().set_ylabel(r'$\theta$')
    elif len(thetas) == 1: 
        plt.figure(figsize=(3.45, 2.5))
        plt.imshow(colors_rgb[:,0,:,:], extent=(400E-9, 700E-9, 0, 1), aspect=(300E-9)/2)
        plt.gca().set_yticks(np.linspace(0,1,3))
        plt.gca().set_yticklabels(np.linspace(1,-1,3))
        plt.gca().set_ylabel(r'$\rho$')
    
    if len(rhos) == 1 or len(thetas) == 1:
        plt.gca().set_xticks(np.linspace(400E-9, 700E-9,4)); plt.gca().set_xticklabels(np.linspace(400,700,4).astype(int))
        plt.gca().set_xlabel('Wavelength $\lambda$ (nm)')
        
    return colors_rgb

def plot_color_shift_cube(n_wavelengths=100, N=10):
    
    xz = color_shift_cube(wavelengths=np.linspace(400E-9, 700E-9, n_wavelengths), rhos=np.linspace(-1,1,N), thetas=np.array([0]), return_XYZ=True)
    xy = color_shift_cube(wavelengths=np.linspace(400E-9, 700E-9, n_wavelengths), rhos=np.array([1]), thetas=np.linspace(0,np.pi,N), return_XYZ=True)
    yz = color_shift_cube(wavelengths=np.array([700E-9]), rhos=np.linspace(-1,1,N), thetas=np.linspace(0,np.pi,N), return_XYZ=True)
    
    max_XZ = np.max(xz[:,:,:,1])
    xz /= max_XZ
    xy /= max_XZ
    yz /= max_XZ
    
    xz = ct.from_xyz_to_rgb(xz, normalize=False)
    xy = ct.from_xyz_to_rgb(xy, normalize=False)
    yz = ct.from_xyz_to_rgb(yz, normalize=False)
    
    canvas = figure(figsize=(3.45, 2.5))
    axes = Axes3D(canvas)
    
    quads = cube(width=1,height=1,depth=1,width_segments=n_wavelengths, height_segments=N, depth_segments=N)
    
    # You can replace the following line by whatever suits you. Here, we compute
    # each quad colour by averaging its vertices positions.
    RGB = np.average(quads, axis=-2)
    RGB[np.average(quads, axis=-2)[:, 0] == 1] = yz.transpose((2,1,0,3)).reshape((-1,3)) 
    RGB[np.average(quads, axis=-2)[:, 2] == 1] = xy.transpose((2,1,0,3)).reshape((-1,3)) 
    RGB[np.average(quads, axis=-2)[:, 1] == 0] = xz.transpose((2,1,0,3)).reshape((-1,3))
    
    # Adding an alpha value to the colour array.
    RGBA = np.hstack((RGB, np.full((RGB.shape[0], 1), 1)))
        
    collection = Poly3DCollection(quads)
    collection.set_color(RGBA)
    axes.add_collection3d(collection)
        
    plt.gca().set_xticks(np.linspace(0,1,4)); plt.gca().set_xticklabels(np.linspace(400,700,4))
    plt.gca().set_xlabel('Wavelength $\lambda$')
    plt.gca().set_yticks([0,0.5,1]); plt.gca().set_yticklabels(['0','$\pi/2$', '\pi'])
    plt.gca().set_ylabel(r'$\theta$')
    plt.gca().set_zticks([-1,0,1])
    plt.gca().set_zlabel(r'$\rho$')
    
    
    return xz, xy, yz
    
    
def plot_light_spectrum(n_wavelengths=100):
    
    lambdas, omegas = generate_wavelengths(n_wavelengths)  
    colors_XYZ = np.zeros((n_wavelengths, 3))
    
    for i, lambd in enumerate( np.linspace(400E-9, 700E-9, n_wavelengths) ):
        print(i)
    #        spectrum = generate_gaussian_spectrum(lambdas=lambdas, mu=lambd, sigma=30E-9)
        spectrum = generate_mono_spectrum(lambdas=lambdas, color=lambd)
        colors_XYZ[i, :] = -ct.from_spectrum_to_xyz(lambdas, spectrum, normalize=False)
        
    #normalize colors
#    colors_xyz = colors_XYZ/np.max(np.sum(colors_XYZ, axis=1))
    colors_xyz = colors_XYZ/np.max(colors_XYZ[:,1])
    colors_rgb = ct.from_xyz_to_rgb(colors_xyz.reshape((1,n_wavelengths,3)), normalize=False)
    colors_rgb /= np.max(colors_rgb, axis=(1,2))[:,None,None]
        
#    plt.figure(figsize=(2*3.45, 2*2.5))
    plt.figure(figsize=(3.45, 2.5))
    plt.imshow(colors_rgb, extent=(400E-9, 700E-9, 0, 1), aspect=(300E-9)/7)
    plt.gca().set_yticks([])
    plt.gca().set_xticks(np.linspace(400E-9, 700E-9,4)); plt.gca().set_xticklabels(np.linspace(400,700,4).astype(int))
    plt.gca().set_xlabel('Wavelength (nm)')
    plt.savefig(fig_path + 'spectrum_light.pdf')
    
    
def load_lippmann(path):
    
    data = np.loadtxt(path)
    lambdas = data[:,0]*1E-9
    spectrum = data[:,1]
    
    return lambdas, spectrum


def show_incident_angles_measurements(path, root, visible=True):
    
    f, axes = plt.subplots(1,4, figsize=(3.45/0.5, 3.45/3.8/0.5))
    
    im = plt.imread(path + 'image.jpg')
    axes[0].imshow(im)
    axes[0].axis('off')
    axes[0].set_title('(a) Original plate')
    
    peaks = np.array(['(b) 531 nm', '(c) 547 nm', '(d) 566 nm'])
    for i, suffix in enumerate(['top_8', 'mid_16', 'bot_16']):
        lambdas, spectrum = load_lippmann(path + root + suffix)
#        if vmax is None:
        vmax = 1.1*np.max(spectrum)
        
        show_spectrum(lambdas, spectrum, ax=axes[i+1], visible=visible, show_background=True, short_display=True, lw=1, vmax=vmax)
        axes[i+1].set_title(peaks[i])
        
#        plt.figure()
#        plt.plot(lambdas, spectrum)
        
    f.tight_layout()
    f.subplots_adjust(hspace=-1.0)
    plt.savefig(fig_path + 'incident_angles.pdf')

def show_developers(path, visible=True, unnormalize=True):
    
    f, axes = plt.subplots(6, 4, figsize=(3.45/0.6, 3.45/0.6*6/4), sharex=True)
    
    for i, letter in enumerate(['A', 'B', 'C', 'D', 'E', 'F']):
        for j, (dev, exp) in enumerate(zip(['', 'Y_', 'YB_', 'L_'], ['4096', '4096', '4096', '512'])):
            lambdas, spectrum = load_lippmann(path + 'cc_' + dev + letter + '3_'+ exp)
            
            if j == 0:
                white = load_lippmann(path + 'spectral-references/white_reference_204_1_'+ exp + '0')[1]
                black = load_lippmann(path + 'spectral-references/black_reference_204_1_'+ exp + '0')[1]
                spectrum *= white
#                spectrum += black
            show_spectrum(lambdas, spectrum, ax=axes[i,j], visible=visible, show_background=True, short_display=True, lw=1)
    
    axes[0,0].set_title('Original')
    axes[0,1].set_title('Ultimate')
    axes[0,2].set_title('Ultimate + bleach')
    axes[0,3].set_title('Lumi\`ere')
    
    plt.savefig(fig_path + 'developers.pdf')
    
def show_developers2(path, visible=True):
    
    f, axes = plt.subplots(6, 6, figsize=(3.45/0.6, 3.45/0.6), sharex=True)
    
    for i, letter in enumerate(['A', 'B', 'C', 'D', 'E', 'F']):
        for j in range(1,4):
            
            lambdas, spectrum = load_lippmann(path + 'cc_' + letter + str(j) + '_4096')
            white = load_lippmann(path + 'spectral-references/white_reference_204_1_40960')[1]
            black = load_lippmann(path + 'spectral-references/black_reference_204_1_40960')[1]
            spectrum *= white
            show_spectrum(lambdas, spectrum, ax=axes[i,j-1], visible=visible, show_background=True, short_display=True, lw=1)
            
            lambdas, spectrum = load_lippmann(path + 'cc_L_' + letter + str(j) + '_1024')
            show_spectrum(lambdas, spectrum, ax=axes[i,2+j], visible=visible, show_background=True, short_display=True, lw=1)
            
    axes[0,0].set_title('Original')
    axes[0,1].set_title('Original')
    axes[0,2].set_title('Original')
    axes[0,3].set_title('Lumi\`ere')
    axes[0,4].set_title('Lumi\`ere')
    axes[0,5].set_title('Lumi\`ere')
   
    plt.savefig(fig_path + 'developers_all.pdf')


def show_gonio_measurements(root, files, visible=False):
    
    for file in files:
        lambdas, spectrum = load_lippmann(root + file)
        plt.figure(figsize=(3.45, 2.5))
        show_spectrum(lambdas, spectrum, ax=plt.gca(), visible=visible, show_background=True) 
#        plt.title(file)
        plt.savefig(fig_path + file + '_spectrum.pdf')
        
        depths = generate_depths(max_depth=7E-6)
        if 'Hg' in file:
            lippmann, _ = lippmann_transform(lambdas, spectrum, depths, r=0.7*np.exp(1j*np.deg2rad(-148)))
        else:   
            lippmann, _ = lippmann_transform(lambdas, spectrum, depths, r=0.2)
        
        plt.figure(figsize=(3.45, 2.5))
        show_lippmann_transform(depths, lippmann, ax=plt.gca()) 
#        plt.title(file)
        plt.savefig(fig_path + file + '_lippmann.pdf')
        

def show_2_gonio_measurements(root, files, visible=False):
            
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.45/2, 3.45/4/335*406))
    lambdas, spectrum = load_lippmann(root + files[0])
    show_spectrum(lambdas, spectrum, ax=ax1, visible=visible, show_background=True, lw=1) 
    
    
    lambdas, spectrum = load_lippmann(root + files[1])
    show_spectrum(lambdas, spectrum, ax=ax2, visible=visible, show_background=True, lw=1) 
    
    plt.savefig(fig_path + 'mercury_vs_air_spectrum.pdf')

def lippmann_phase_retrieval(root, files, N_depth=93, Z=5E-6, visible=True):
    
    lambdas, lipp_glass = load_lippmann(root + files[0])
    _, lipp = load_lippmann(root + files[1])
    _, glass = load_lippmann(root + files[2])
    
    lipp_glass = np.sqrt(lipp_glass)
    lipp = np.sqrt(lipp)
    glass = np.sqrt(glass)
    
    #use the law of cosines
    print(glass, lipp, lipp_glass)
    print(np.max(lipp_glass))
    vmax = np.max(lipp_glass)*1.1
    
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(3.45, 3.45/3))
    show_spectrum(lambdas, lipp_glass, ax=ax1, visible=visible, show_background=True, lw=1, vmax=vmax) 
    show_spectrum(lambdas, lipp, ax=ax2, visible=visible, show_background=True, lw=1, vmax=vmax) 
    show_spectrum(lambdas, glass, ax=ax3, visible=visible, show_background=True, lw=1, vmax=vmax) 
    
    
    phases = np.pi - np.arccos( ( glass**2 + lipp**2 - lipp_glass**2 )/(2*glass*lipp) )
    phases = np.nan_to_num(phases)
    
    plt.figure()
    plt.plot(lambdas, np.mod(phases, 2*np.pi))
    show_spectrum(lambdas, np.mod(phases, 2*np.pi), visible=visible, show_background=True, lw=1) 
    
    c=299792458/1.5
    omegas = 2 * np.pi * c / lambdas 
    depths = np.linspace(0,Z*(1-1/N_depth),N_depth)
    
    #restrict to visible
    idx = np.where((lambdas <= 700E-9) & (lambdas >= 390E-9))
#    idx = np.arange(len(lambdas))
    
    lipp_complex = lipp[idx]*np.exp(-1j*phases[idx])
    A = get_sampling_matrix(omegas[idx], depths, c)
        
    print(np.linalg.matrix_rank(A))
    
    density = np.linalg.lstsq(np.vstack([np.real(A),np.imag(A)]), np.hstack([np.real(lipp_complex), np.imag(lipp_complex)]))[0]
    plt.figure()
    plt.plot(depths, density)
    show_lippmann_transform(depths, density) 
    
    print(density)
    
    show_spectrum(lambdas[idx], np.abs(A @ density)**2, visible=visible, show_background=True, lw=1, vmax=vmax) 


def get_sampling_matrix(omegas, depths, c):
    
    return np.exp(-2j*omegas[:,None]*depths[None,:]/c)


def show_hyspex_measurements(visible=True):
            
#    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(3.45/0.6, 3.45/2.5/0.6))
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(3.45/0.7, 3.45/2.5/0.7))
#    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(3.45*1.4, 3.45/2.5*1.4)) #PNAS
    rgb = np.load('Data/hyspex_rgb.npy')
    green = np.load('Data/hyspex_green.npy')
    red = np.load('Data/hyspex_red.npy')
    w = np.load('Data/hyspex_wavelengths.npy')
    
    ax1.imshow(rgb, aspect='auto')
    ax1.axis('off')
    ax1.set_xlabel('(a) Lippmann plate')

    print(w)

    show_spectrum(w, green, ax=ax2, visible=visible, show_background=True, lw=1, short_display=True) 
    show_spectrum(w, red, ax=ax3, visible=visible, show_background=True, lw=1, short_display=True) 
    ax1.set_title('(a) Lippmann plate')
    ax2.set_title('(b) green')
    ax3.set_title('(c) orange')
    
    ax1.set(adjustable='box-forced')
    ax2.set(adjustable='box-forced')
    ax3.set(adjustable='box-forced')
    
    f.tight_layout()
    f.subplots_adjust(hspace=-1.0)
    
    plt.savefig(fig_path + 'hyspex.pdf')


def show_fib_PNAS():
    
    Z1 = 5.639E-6
    Z2 = 7.75E-6
    f = plt.figure(figsize=(3.45*1.4, 3.45/2.7*1.4)) #PNAS
    
    gs = gridspec.GridSpec(1, 3, width_ratios=[4, 1, 1])
    
    im = plt.imread("/Users/gbaechle/EPFL/PhD/BRDF Data and Rendering Engines/CIME/Lippmann4_07_PNAS.jpg")
    im2 = plt.imread("/Users/gbaechle/EPFL/PhD/BRDF Data and Rendering Engines/CIME/Lippmann4_07_PNAS_2.jpg")
    
    average = np.mean(im2, axis=1)
    
    depths = np.linspace(0, Z2, len(average))
    
    ax1 = plt.subplot(gs[0, 0])
    ax1.imshow(im, cmap='gray', extent=(0, 1, 5.639, -0.577), aspect='auto')
    set_axis_white(ax1)
    ax1.set_xticks([])
    ax1.set_ylabel('Depth')
    
    ax2 = plt.subplot(gs[0, 1], sharey=ax1)
    ax2.imshow(average.reshape((-1,1)), cmap='gray', extent=(0, 1, 5.639, -(Z2/Z1-1)*5.639), vmax=np.max(average), aspect='auto')
    set_axis_white(ax2)
    ax2.set_xticks([])
    ax2.set_yticks([])
    
    rho = 0.7
    theta = -148
    r = rho*np.exp(1j*np.deg2rad(theta))
    lambd = 531E-9
    c = 299792458/1.5*10
    omega = 2*np.pi*c/lambd
    
    depths = np.linspace(0, Z2, len(average))
    theoretical = 1+rho**2+2*rho*np.cos(2*omega*depths/c + theta)
    ax3 = plt.subplot(gs[0, 2], sharey=ax1)
    ax3.imshow(theoretical.reshape((-1,1)), cmap='gray', extent=(0, 1, 5.639, -(Z2/Z1-1)*5.639), vmin=0, vmax=np.max(theoretical), aspect='auto')
    set_axis_white(ax3)
    ax3.set_xticks([])
    ax3.set_yticks([])
    
    ax1.set_xlabel('(a)')
    ax2.set_xlabel('(b)')
    ax3.set_xlabel('(c)')

    
    plt.savefig(fig_path + 'fib.pdf')
    
def show_fib_PNAS2():
    
    Z1 = 5.639E-6
    Z2 = 7.75E-6
    f = plt.figure(figsize=(3.45*1.4, 3.45/2.7*1.4)) #PNAS
    
    gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1])
    
    im = plt.imread("/Users/gbaechle/EPFL/PhD/BRDF Data and Rendering Engines/CIME/Lippmann4_07_PNAS.jpg")
    
    average = np.mean(im, axis=1)
    
    depths = np.linspace(0, Z2, len(average))
    
    ax1 = plt.subplot(gs[0, 0])
    ax1.imshow(im, cmap='gray', extent=(0, 1, 5.639, -0.577), aspect='auto')
    set_axis_white(ax1)
    ax1.set_xticks([])
    ax1.set_ylabel('Depth ($\mu m$)')
    
    ax2 = plt.subplot(gs[0, 1])
    ax2.imshow(average.reshape((-1,1)), cmap='gray', extent=(0, 1, 5.639, -0.577), vmax=0.5*np.max(average), aspect='auto')
    set_axis_white(ax2)
    ax2.set_xticks([])
    ax2.set_yticks([])
   
    ax1.set_xlabel('(a) Electron microscope image')
    ax2.set_xlabel('(b) Average')

    
    plt.savefig(fig_path + 'fib2.pdf')
    
def set_axis_white(ax):
    
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white') 
    ax.spines['right'].set_color('white')
    ax.spines['left'].set_color('white')
    
    
def sampling_experiment(Z=5E-6, N=200):
    
    lambdas, omegas = generate_wavelengths(N) #3000
    depths = generate_depths(max_depth=Z)
    
    spectrum = generate_gaussian_spectrum(lambdas=lambdas, mu=GREEN, sigma=20E-9)
#    spectrum = generate_gaussian_spectrum(lambdas, mu=420E-9, sigma=30E-9) + generate_gaussian_spectrum(lambdas, mu=550E-9, sigma=40E-9) + 0.5*generate_gaussian_spectrum(lambdas, mu=500E-9, sigma=8E-9) + 0.1*(1.5+np.cos(lambdas*c/2))
#    spectrum = generate_mono_spectrum(lambdas=lambdas, color=530E-9)

    A  = 4/np.pi*np.abs(omegas[1]-omegas[0])/c*np.real(fda.get_sampling_matrix(omegas, Z, mode=1))
    Ai = 4/np.pi*np.abs(omegas[1]-omegas[0])/c*np.imag(fda.get_sampling_matrix(omegas, Z, mode=1))
#    A *= np.linalg.norm(spectrum)/np.linalg.norm(A@spectrum)
    
    
#    F = fda.generate_matrix_F(omegas, Z) 
#    F = fda.generate_matrix_F2(omegas, Z) 
    F = 4/np.pi*np.abs(omegas[1]-omegas[0])/c*np.real(fda.get_sampling_matrix(omegas, Z, mode=2))
#    F *= np.linalg.norm(spectrum)/np.linalg.norm(A @ spectrum)
#    F *= np.linalg.norm(A)*np.linalg.norm(F)
    B = A
 
    u,s,vh = np.linalg.svd(B)
    
    idx = np.where(s>0.1)[0]
    u_ = u[:,idx]
    vh_ = vh[idx,:]

    Bpinv = np.conj(vh_).T @ np.diag(1/s[idx]) @ np.conj(u_).T
    
    new_spectrum = A @ spectrum
    new_spectrum += np.random.random(len(spectrum))*0.2
    
#    rec = np.linalg.lstsq(A, new_spectrum)[0]
#    rec = invert_sampling_operation(A, new_spectrum, omegas, Z)
#    rec = fda.invert_G(omegas, Z, A) @ new_spectrum
    
#    rec = F @ Bpinv @ new_spectrum
    rec = Bpinv @ new_spectrum
#    rec = F @ rec
    
    
    freqs = np.fft.fftshift(np.fft.fftfreq(N, d=2*(omegas[0]-omegas[1])/c))
#    plt.figure(); plt.plot(freqs, np.real(np.fft.fftshift(np.fft.fft(spectrum))))
    show_lippmann_transform(depths, lippmann_transform(lambdas, spectrum, depths)[0])
    plt.gca().set_yticks([0])
    show_spectrum(lambdas, spectrum); plt.title('original')
#    show_spectrum(lambdas, new_spectrum, ax=plt.gca())
    show_spectrum(lambdas, new_spectrum); plt.title('transformed')
    show_spectrum(lambdas, Ai @ spectrum); plt.title('transformed (imag)')
    show_spectrum(lambdas, rec); plt.title('reconstruction')
    show_spectrum(lambdas, np.real(fda.get_sampling_matrix(omegas, Z, mode=2)) @ rec); plt.title('reconstruction (lowpass)')
    
    density = lippmann_transform(lambdas, rec, depths)[0]
    show_lippmann_transform(depths, density)
    plt.gca().set_yticks([0])
    show_spectrum(lambdas, lippmann_transform_reverse(lambdas, density, depths)); plt.title('inverse of reconstructed density')
    
    plt.figure()
    plt.plot(depths, lippmann_transform(lambdas, spectrum, depths)[0], '--')
    plt.plot(depths, density, '--')
#    depths = generate_depths(max_depth=2*Z)
#    density = lippmann_transform(lambdas, rec, depths)[0]
#    show_lippmann_transform(depths, density)
    
    
#    plt.figure(); plt.plot(freqs, np.abs(np.fft.fftshift(np.fft.fft(rec))))
    show_spectrum(lambdas, A @ rec); plt.title('consistency')
    show_spectrum(lambdas, Ai @ rec); plt.title('consistency (imag)')
    
    
def invert_sampling_operation(A, y, omegas, Z, n_iter=1000):
    
    yi = np.copy(y)
    for i in range(n_iter):
        yi += y - A @ yi
#        yi = np.real(fda.get_sampling_matrix(omegas, Z, mode=2)) @ yi
#        yi = fda.project_on_bandlimited_space3(omegas, yi, Z)
#    yi = np.real(fda.get_sampling_matrix(omegas, Z, mode=2)) @ yi 
    return yi
        
    

if __name__ == '__main__':
    
#    plt.close('all')
    
    RED = 665E-9
    ORANGE = 600E-9
    YELLOW = 580E-9
    GREEN = 550E-9
    CYAN = 480E-9
    INDIGO = 470E-9
    VIOLET = 460E-9
    
    lambdas, omegas = generate_wavelengths(3000) #3000
    depths = generate_depths(max_depth=5E-6)
    
#    spectrum = generate_gaussian_spectrum(lambdas=lambdas, mu=GREEN, sigma=30E-9)
    spectrum = generate_mono_spectrum(lambdas=lambdas, color=530E-9)
    
    lippmann_hg, _ = lippmann_transform(lambdas, spectrum, depths, r=-0.7)
    lippmann_air, _ = lippmann_transform(lambdas, spectrum, depths, r=0.2)
#    window = (np.cos(np.linspace(0, np.pi, len(depths)))+1)/2
#    window = np.linspace(1,0, len(depths))
#    window = np.exp(np.linspace(0, -7, len(depths)))
    window = erfc(np.linspace(0,4,len(depths))) 
#    window = np.ones(len(depths))
    lippmann_hg  *= window
    lippmann_air *= window
    
    omegas = 2 * np.pi * c / lambdas
        
    spectrum_hg = inverse_lippmann(lippmann_hg, lambdas, depths)
    spectrum_air = inverse_lippmann(lippmann_air, lambdas, depths)
    
    c1 = ct.from_xyz_to_rgb( ct.from_spectrum_to_xyz(lambdas, spectrum).reshape((1,1,-1)) )
    c2 = ct.from_xyz_to_rgb( ct.from_spectrum_to_xyz(lambdas, spectrum_hg).reshape((1,1,-1)) )
    c3 = ct.from_xyz_to_rgb( ct.from_spectrum_to_xyz(lambdas, spectrum_air).reshape((1,1,-1)) )
    
#    plt.close('all')    
    
#    plt.figure()
#    show_spectrum(lambdas, spectrum, ax=plt.gca(), show_background=False) 
#    plt.title('original spectrum') 
#    plt.savefig(fig_path + 'original.pdf')
#    
#    plt.figure()
#    show_lippmann_transform(depths, lippmann_hg, ax=plt.gca()) 
#    plt.title('Lippmann transform Hg')
#    plt.savefig(fig_path + 'lippmann_hg.pdf')
#    
#    plt.figure()
#    show_lippmann_transform(depths, lippmann_air, ax=plt.gca()) 
#    plt.title('Lippmann transform Air')
#    plt.savefig(fig_path + 'lippmann_air.pdf')
#    
#    plt.figure()
#    show_spectrum(lambdas, spectrum_hg, ax=plt.gca(), show_background=False) 
#    plt.title('Spectrum Hg (inverse Lippmann)') 
#    plt.savefig(fig_path + 'hg.pdf')
#    
#    plt.figure()
#    show_spectrum(lambdas, spectrum_air, ax=plt.gca(), show_background=False) 
#    plt.title('Spectrum Air (inverse Lippmann)') 
#    plt.savefig(fig_path + 'air.pdf')
#    
#    plt.figure();
#    plt.imshow( np.r_[c1, c2, c3])
#    plt.savefig(fig_path + 'colors.pdf')
#    
#    root = '/Users/gbaechle/EPFL/PhD/BRDF Data and Rendering Engines/Goniophotometer/2018-05-01/'
#    files = ['lippmann_01_05_air_64', 'lippmann_01_05_hg_64', 'lippmann_01_05_air_swollen_64', 'lippmann_01_05_air_prism_64', 'lippI_air1', 'lippI_hg1', 'lippmann_01_05_glass']
#    files = ['lippmann_01_05_hg_64', 'lippmann_01_05_hg_prism_64', 'lippmann_01_05_glass_64']
#    files = ['lippmann_01_05_air_64', 'lippmann_01_05_air_prism_64', 'lippmann_01_05_glass_64']


#    show_gonio_measurements(root, files, visible=False)

#    lippmann_phase_retrieval(root, files)

    root = '/Users/gbaechle/EPFL/PhD/BRDF Data and Rendering Engines/Goniophotometer/'
    files = ['lipp22', 'lipp28', 'lippI_air1', 'lippI_hg1']

#    show_gonio_measurements(root, files, visible=True)
#    show_gonio_measurements(root, files, visible=True)
#    show_2_gonio_measurements(root, files[:2], visible=True)
#    show_hyspex_measurements()
    
    #measured intensities
#    plot_fib("/Users/gbaechle/EPFL/PhD/BRDF Data and Rendering Engines/CIME/", "Lippmann4_07_hc.jpg", 5.55E-6, n0=1.52)
#    plot_fib("/Users/gbaechle/EPFL/PhD/BRDF Data and Rendering Engines/CIME/", "Lippmann4_07_hc.jpg", 7.75E-6, n0=1.52)
#    plot_fib("/Users/gbaechle/EPFL/PhD/BRDF Data and Rendering Engines/CIME/", "ESB stack aligned small (binned)_curves.png", 1.47*5.52E-6, n0=1.47)

#    plot_fibs()
    
    color_shift(n_wavelengths=100)
    plt.savefig(fig_path + 'color_shift2.pdf')
    
#    color_shift_2_figs(n_wavelengths=500, N=100)
    
#    xz, xy, yz = plot_color_shift_cube(n_wavelengths=500, N=100)
#    plt.savefig(fig_path + 'color_shift_cube.pdf')
    
#    plot_light_spectrum(n_wavelengths=1000)
  
#    plt.figure(figsize=(3.45*1.4, 1.7*1.4))
#    color_shift(n_wavelengths=100, ax=plt.gca())
    
#    show_hyspex_measurements(visible=True)
    
#    path = '/Users/gbaechle/EPFL/PhD/BRDF Data and Rendering Engines/Goniophotometer/2018-06-15 viewing_angles/'
#    root = 'viewing_angles_'
#    show_incident_angles_measurements(path, root)
#    
#    path = '/Users/gbaechle/EPFL/PhD/BRDF Data and Rendering Engines/Goniophotometer/2018-06-15 color_checker/'
#    show_developers(path)
    
#    path = '/Users/gbaechle/EPFL/PhD/BRDF Data and Rendering Engines/Goniophotometer/2018-07-11 color_checker/'
#    show_developers2(path)
    
#    sampling_experiment()
    
#    show_fib_PNAS2()
    
    
    
    