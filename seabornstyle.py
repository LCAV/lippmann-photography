# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 14:01:34 2017

@author: gbaechle
"""


import seaborn as sns
import matplotlib as mpl

def setStyleMinorProject(mode='inColumn', black=False) :
	
    sns.set_context("paper", font_scale=1.0, rc={"lines.linewidth": 1.0})

    font = {'family' : 'serif',
            'serif'  : ['Times New Roman'],
            'sans-serif'  : ['Times New Roman'],
            'weight' : 'normal',
            'size'   : 8}

    mpl.rc('font', **font)
 
    sns.set_style("white")
 
    if black:
        sns.set(rc={'axes.facecolor':'black', 'axes.edgecolor':'white', 'figure.facecolor': 'black'})
	
    mpl.pyplot.rcParams['legend.facecolor'] = (1, 1, 1)
    mpl.pyplot.rcParams['legend.frameon'] = False
    mpl.pyplot.rcParams['legend.framealpha'] = 0.5
    mpl.pyplot.rcParams['legend.loc'] = 'best'
 
#	mpl.pyplot.rcParams['axes.facecolor'] = '0.95'
	
    #mpl.pyplot.rcParams['figure.figsize'] = 3.28, 2.36 #this is the size of a column, in inch
    mpl.pyplot.rcParams['figure.figsize'] = 3.28, 2. #this is the size of a column, in inch
    mpl.pyplot.rcParams['figure.dpi'] = 72 #screen
    mpl.pyplot.rcParams['figure.autolayout'] = True #screen
	
    mpl.pyplot.rcParams['savefig.format'] = 'pdf' #this is the size of a column, in inch
    mpl.pyplot.rcParams['savefig.dpi'] = 300 #neat
	
    #sns.set_palette("husl")
    sns.set_palette("Blues_r")
    

def setStylePNAS():
	
    sns.set_context("paper", font_scale=1.0, rc={"lines.linewidth": 0.5})
    
    sns.set(font="Helvetica")
    
    sns.set_style("white")
 
    mpl.pyplot.rcParams['legend.facecolor'] = (1, 1, 1)
    mpl.pyplot.rcParams['legend.frameon'] = False
    mpl.pyplot.rcParams['legend.framealpha'] = 0.5
    mpl.pyplot.rcParams['legend.loc'] = 'best'
 
#	mpl.pyplot.rcParams['axes.facecolor'] = '0.95'
	
    #mpl.pyplot.rcParams['figure.figsize'] = 3.28, 2.36 #this is the size of a column, in inch
    mpl.pyplot.rcParams['figure.figsize'] = 3.28, 2. #this is the size of a column, in inch
    mpl.pyplot.rcParams['figure.dpi'] = 72 #screen
    mpl.pyplot.rcParams['figure.autolayout'] = True #screen
	
    mpl.pyplot.rcParams['savefig.format'] = 'pdf' #this is the size of a column, in inch
    mpl.pyplot.rcParams['savefig.dpi'] = 300 #neat
    
    mpl.rcParams['lines.linewidth'] = 1
    
    font = {'family' : 'Helvetica',
            'sans-serif'  : ['Helvetica'],
            'weight' : 'normal',
            'size'   : 8}

    mpl.pyplot.rc('font', **font)
    
  
	
 