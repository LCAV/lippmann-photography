import matplotlib.pyplot as plt
import numpy as np

import holopy as hp
from holopy.scattering import calc_scat_matrix, Sphere

plt.close('all')

silver_index = 1.74
gelatin_index = 1.54
grain_radius = 0.01       #in micrometers

illum_wavelen = 0.460
illum_polarization = (1,0)

n_points = 1000
angles = np.linspace(-np.pi, np.pi, n_points)

detector = hp.detector_points(theta=angles, phi=0)
distant_sphere = Sphere(r=grain_radius, n=silver_index)
matr = calc_scat_matrix(detector, distant_sphere, gelatin_index, illum_wavelen)

#cartesian plot
plt.figure()
plt.semilogy(angles, abs(matr[:,0,0])**2)
plt.semilogy(angles, abs(matr[:,1,1])**2)
plt.gca().set_xlim([angles[0], angles[-1]])
plt.show()


#polar plot
plt.figure()
plt.subplot(111, projection='polar')
plt.plot(angles, abs(matr[:,0,0])**2)
plt.plot(angles, abs(matr[:,1,1])**2)
plt.gca().set_rmax(np.max(abs(matr[:,0,0])**2))
plt.gca().set_rticks([])
plt.show()