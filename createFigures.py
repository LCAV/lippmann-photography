# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 14:01:34 2017

@author: gbaechle
"""


import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
from lippmann import *

import seaborn as sns
import seabornstyle as snsty

snsty.setStyleMinorProject()

plt.close("all")

def remove_spines(ax):
    
    ax.spines["top"].set_visible(False)        
    ax.spines["right"].set_visible(False)    
    ax.spines["left"].set_visible(False)       
    ax.spines['bottom'].set_linewidth(0.4)
    ax.spines['bottom'].set_color('0.2')   

def add_horizontal_lines(xrange, yrange):
    for y in yrange:    
        plt.plot(xrange, [y] * len(xrange), ":", lw=0.6, color="black", alpha=0.1) 



if __name__ == '__main__':
    
    plot_gaussian_lippmann_and_inverses()
    plot_mono_lippmann_and_inverses()
    plot_rect_lippmann_and_inverses()



