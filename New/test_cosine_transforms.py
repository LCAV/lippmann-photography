#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 15:18:17 2018

@author: gbaechle
"""
import numpy as np
import scipy as sp

import matplotlib.pyplot as plt

c = 299792458
#c = 2

def cosine_transform(x, y, w, inv=False, theta=0):
    
    cosines = np.cos(x[None, :] * w[:, None]/c + theta)
    if inv:
        return 2/np.pi*np.trapz(cosines * y[None, :], x, axis=1)
    else:
        return np.trapz(cosines * y[None, :], x, axis=1)
    
def inverse_cosine(x, y, w, theta=0):
    
    nu = 2*np.mod(-theta, 2*np.pi)/np.pi
    integrand = (w[:,None]*x[None,:]/c)**nu * \
                (sp.special.hyp1f1(1,1+nu, 1j*w[:,None]*x[None,:]/c) + \
                 sp.special.hyp1f1(1,1+nu, -1j*w[:,None]*x[None,:]/c))
                
    return 1/(c*np.pi*sp.special.gamma(nu+1)) * np.trapz(integrand * y[:, None], w, axis=0)
    
    
def fourier_transform(x, y, w, inv=False, theta=0):

    if inv:
        exps = np.exp(1j*( x[None, :] * w[:, None] + theta ))
        return 2/np.pi*np.trapz(exps * y[None, :], x, axis=1)
    else:
        exps = np.exp(-1j*( x[None, :] * w[:, None] + theta ))
        return np.trapz(exps * y[None, :], x, axis=1)
    
    
def sine_transform(x, y, w, inv=False, theta=0):
    
    sines = np.sin(x[None, :] * w[:, None] + theta)
    if inv:
        return 2/np.pi*np.trapz(sines * y[None, :], x, axis=1)
    else:
        return np.trapz(sines * y[None, :], x, axis=1)
    


def lippmann_transform(x, y, w, theta=0):

    cosines = np.cos(x[None, :] * w[:, None] - theta)
    return np.trapz(cosines * y[None, :], w, axis=1)
    

def plt_complex(x,y,ax=None):
    
    if ax is None:
        plt.figure()
        ax = plt.gca()
        
    ax.plot(x,np.real(y))
    ax.plot(x,np.imag(y), ':')


if __name__ == '__main__':
    
    plt.close('all')
    
    N = 1000
    theta = np.pi/3
    
    x = np.linspace(0,N,N)
    w = np.linspace(0,c/10,N)
    x_sym = np.linspace(-N,N,2*N-1)
    w_sym = np.linspace(-1/10, 1/10, 2*N-1)
    
    y = sp.stats.norm(loc=N/3, scale=50).pdf(x)
    y_e = sp.stats.norm(loc=N/3, scale=50).pdf(x_sym) + sp.stats.norm(loc=-N/3, scale=50).pdf(x_sym)
    y_o = sp.stats.norm(loc=N/3, scale=50).pdf(x_sym) - sp.stats.norm(loc=-N/3, scale=50).pdf(x_sym)
    
    y_f = np.cos(theta)*y_e + np.sin(theta)/1j*y_o
    
    z = cosine_transform(x, y, w, theta=theta)
    z_s = np.gradient(z, w[1])
#    z_s = sp.misc.derivative(z, w[1])
    z_f = z -1j*z_s
#    z_f[w==0] = z[0] # correct for w = 0
    
    
    y_magic = inverse_cosine(x, z, w, theta=theta)
    plt.figure(); plt.plot(x,y); plt.plot(x, y_magic, 'r:')
    
    plt.figure(); plt.plot(w,z); plt.plot(w,sine_transform(x, y, w, theta=theta), 'r:')
    plt.figure(); plt.plot(w,np.real(z_f)); plt.plot(w,np.imag(z_f), 'r:')
    z_f *= np.exp(-1j*theta)


    z_sym = np.r_[np.conj(z[:0:-1]), z]
    y2 = cosine_transform(w, z, x, inv=True, theta=theta)
    
    zf = fourier_transform(x_sym, y_sym, w_sym, theta=theta)
    y3 = fourier_transform(w_sym, z_sym, x_sym, inv=True, theta=0)
    z2 = np.cos(theta)*cosine_transform(x, y, w) + np.sin(theta)/1j*sine_transform(x, y, w)
    z3 = fourier_transform(x_sym, y_f, w_sym)
    
    plt.figure(); plt.plot(w,np.real(z_f)); plt.plot(w,np.imag(z_f), 'r:')
    plt.figure(); plt.plot(w_sym,zf); plt.plot(w_sym,np.imag(zf), 'r:')
    
    plt.figure(); plt.plot(x,y); plt.plot(x,y2, 'r:');# plt.plot(x_sym,y3/(np.cos(theta)+1/1j*np.sin(theta)), 'g--')
    plt.figure(); plt.plot(x,y); plt.plot(x_sym, y_f, 'r:'); plt.plot(x_sym, np.imag(y_f), 'g--')
    
    plt.figure(); plt.plot(w_sym,z_sym); plt.plot(w_sym,zf/2, 'r:'); plt.plot(w_sym, z3/2, 'g--')
    
    plt_complex(w_sym, z3/2)
    plt_complex(w_sym, zf/2)
    
    
    
    
    