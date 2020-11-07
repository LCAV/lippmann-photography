# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 13:39:13 2017

@author: gbaechle
"""

import numpy as np
import scipy as sp
import scipy.optimize

N = 100

H = np.random.rand(N, N) + 1j * np.random.rand(N, N)
Hr = np.real(H)
Hi = np.imag(H)

p = np.random.rand(N)

i = np.abs(H @ p)**2

#print(i)
#print((Hr @ p)**2 + (Hi @ p)**2)

p_est = np.random.rand(N)


for n in range(10):
    
    p_est =  np.linalg.lstsq( Hr, np.sqrt( i-(Hi @ p_est)**2 ) )[0]
    print(p_est)
    p_est =  np.linalg.lstsq( Hi, np.sqrt( i-(Hr @ p_est)**2 ) )[0]
    print(p_est)
    
print('------------')
print(p)

diff = lambda x: (Hr @ x)**2 + (Hi @ x)**2 - i
diff = lambda x: np.abs(H @ x)**2 - i
p_est = sp.optimize.least_squares(diff, np.random.rand(N)).x

print(p_est)



    
    
    
    
    