# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from color_tools import *

#from lippmann import LippmannPlate


class GuiManager(object):
    
    def __init__(self, lippmann, normalize_spectrums=False, gamma_correct=False):
        self.lippmann = lippmann
        
        if normalize_spectrums:
            self.spectrum_max_value = np.max(lippmann.spectrums.intensities)
            self.spectrum_new_max_value = np.max(lippmann.new_spectrums.intensities)
#            self.spectrum_new_max_value = 0.
        else:
            self.spectrum_max_value = 0.
            self.spectrum_new_max_value = 0.
            
        self.gamma_correct = gamma_correct
        
        self.image_ax = None
        self.spectrum_ax = None
        self.spectrogram_ax = None

        self.new_spectrum_ax = None
        self.new_image_ax = None

        self.fig_number = None 
        self.pixel_index = 0
        self.pixel_index_2D = [0,0]
        
        self.gs = gridspec.GridSpec(2, 3,
                         height_ratios = [10,10],
                         width_ratios = [3,3,1])
                         
        self.lim_x = None

    #show the figures        
    def show(self):
        fig = plt.figure()
        fig.patch.set_facecolor('white')
        self.fig_number = fig.number   
        
        #first show the clickable image
        self.plot_image()    
        
        #add listeners
        fig.canvas.mpl_connect('button_press_event', self.on_click)
        fig.canvas.mpl_connect('key_press_event', self.key_pressed)
        
        #show the reflectance
        self.plot_reflectance()
        
        #show the spectrum
        self.plot_spectrums()
        
        #show the reconstructed image
        self.plot_new_image()
        
        plt.show()
        
        
    def on_click(self, event):
        if event.xdata != None and event.ydata != None:
            print(event.xdata, event.ydata)

            y = int(event.ydata)
            x = int(event.xdata)
                        
            if event.inaxes == self.image_ax or event.inaxes == self.new_image_ax:       
                
                n_y = self.lippmann.height
                self.pixel_index = y + x*n_y
                self.pixel_index_2D = [y, x]
                
                self.plot_image()
                self.plot_spectrums()
                self.plot_reflectance()
                self.plot_new_image()
                
    def key_pressed(self, event):
        n_x = self.lippmann.width
        n_y = self.lippmann.height
        x = self.pixel_index_2D[1]
        y = self.pixel_index_2D[0]

        if event.key== 'up' and y > 0:
                y -= 1
        elif event.key== 'down' and y < n_y:
                y += 1
        elif event.key== 'left' and x > 0:
                x -= 1
        elif event.key== 'right' and x < n_x:
                x += 1 
 
        self.pixel_index = y + x*n_y
        self.pixel_index_2D = [y, x]

        #update the plots        
        self.plot_image()
        self.plot_spectrums()
        self.plot_reflectance()
        self.plot_new_image()
                
            
    def plot_image(self):
#        image = self.lippmann.rgb_ref
        image = self.lippmann.spectrums.rgb_colors
        
        #gamma correction
        if self.gamma_correct:
            image = np.power(image, 2.2)

        self.image_ax = plt.subplot(self.gs[0,0])
        self.image_ax.clear()
        
        self.image_ax.imshow(image)
        self.image_ax.axes.get_xaxis().set_visible(False)
        self.image_ax.axes.get_yaxis().set_visible(False)
        
        self.draw_circle(self.image_ax)
        
        #show the position
        self.image_ax.set_title('Pixel position: (' + str(self.pixel_index_2D[0]) + ', ' + str(self.pixel_index_2D[1]) + ')')
        
        
    def plot_new_image(self):

        image = self.lippmann.new_spectrums.rgb_colors
                
        #gamma correction
        if self.gamma_correct:
            image = np.power(image, 2.2)

        self.new_image_ax = plt.subplot(self.gs[1,0])
        self.new_image_ax.clear()
        
        self.new_image_ax.imshow(image)
        self.new_image_ax.axes.get_xaxis().set_visible(False)
        self.new_image_ax.axes.get_yaxis().set_visible(False)
        
        self.draw_circle(self.new_image_ax)
        
        #title
        self.new_image_ax.set_title('Reconstructed image')
                
        
    def draw_circle(self, ax):
        
        #draw circle
        circle_radius = np.minimum(self.lippmann.height, self.lippmann.width)/50.0
        circle = plt.Circle((self.pixel_index_2D[1], self.pixel_index_2D[0]), circle_radius, color='r', alpha=0.5)
        ax.add_artist(circle)
            
    
    def plot_reflectance(self):

        z           = self.lippmann.r[:,2]
        reflectance = self.lippmann.reflectances[self.pixel_index_2D[0], self.pixel_index_2D[1],:]
        
        self.spectrogram_ax = plt.subplot(self.gs[:,2])
        self.spectrogram_ax.clear()
        
        if self.lim_x is None:
            self.lim_x = np.max( self.lippmann.reflectances)      
        
        self.spectrogram_ax.plot(reflectance, z)
        self.spectrogram_ax.set_ylim([0., np.max(z)])
        self.spectrogram_ax.set_xlim([0, self.lim_x])
        self.spectrogram_ax.set_xticks([0, self.lim_x])
        
        self.spectrogram_ax.set_title('Lippmann transform')

        
    def plot_spectrums(self):
        
        spectrum = self.lippmann.spectrums.get_spectrum(self.pixel_index_2D[0], self.pixel_index_2D[1])
        self.spectrum_ax = plt.subplot(self.gs[0,1])
        spectrum_rgb = self.lippmann.spectrums.rgb_colors[self.pixel_index_2D[0], self.pixel_index_2D[1], :]
        title = 'Spectrum (RGB = [' + format(spectrum_rgb[0], '.2f') + ', ' + format(spectrum_rgb[1], '.2f') + ', ' + format(spectrum_rgb[2], '.2f') + '])'
        spectrum.show(ax=self.spectrum_ax, title=title, y_max=self.spectrum_max_value)
        
        new_spectrum = self.lippmann.new_spectrums.get_spectrum(self.pixel_index_2D[0], self.pixel_index_2D[1])
        self.new_spectrum_ax = plt.subplot(self.gs[1,1])
        new_spectrum_rgb = np.sqrt(self.lippmann.new_spectrums.rgb_colors[self.pixel_index_2D[0], self.pixel_index_2D[1], :])        
        new_title = 'Spectrum new (RGB = [' + format(new_spectrum_rgb[0], '.2f') + ', ' + format(new_spectrum_rgb[1], '.2f') + ', ' + format(new_spectrum_rgb[2], '.2f') + '])'   
        new_spectrum.show(ax=self.new_spectrum_ax, title=new_title, sqrt=True, y_max=self.spectrum_new_max_value)        
        
    

