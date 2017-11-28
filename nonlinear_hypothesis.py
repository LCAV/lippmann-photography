from printing_simulations import *
import numpy as np
import matplotlib.pyplot as plt


N = 1000
n0 = 1.45
dn = 1e-10
saturation = 0.6
cut = 200e-9
c0 = 299792458
c = c0 / n0
N_omegas = 300
delta_z = 10E-9
cut_idx = math.ceil(cut/delta_z)
max_depth = 5E-6
lambdas = wavelengths_omega_spaced()

spectrum = generate_gaussian_spectrum(lambdas=lambdas, mu=550E-9, sigma=50E-9)
# spectrum = generate_mono_spectrum(lambdas)
# spectrum = generate_rect_spectrum(lambdas)

plt.figure()
plt.plot(lambdas, spectrum)
plt.title('Original object spectrum')

depths = np.arange(0, max_depth, delta_z)
intensity, delta_intensity = lippmann_transform(lambdas / n0, spectrum, depths)

# normalize the the intensity
ns = delta_intensity - np.min(delta_intensity)
ns = ns / (np.max(ns))
# change the power leading to different cut from the saturation
ns_sat = sigmoid(ns/saturation)
ns_sat = ns_sat*dn + n0
# below saturation (but with sigmoid)
ns = sigmoid(ns)
ns = ns*dn + n0
# cutting the fist block, (after the saturation, see physics)
ns_cut = ns_sat[cut_idx:]
depths_cut = depths[cut_idx:]



plt.figure()
plt.plot(depths, ns, label="standard (sigmoid)")
plt.plot(depths, ns_sat, label="saturation")
plt.plot(depths_cut, ns_cut, label="saturation + cut")
plt.title('Refractive index')

r, _ = propagation_arbitrary_layers_Born_spectrum(ns, d=delta_z, lambdas=lambdas, plot=False)
r_sat, _ = propagation_arbitrary_layers_Born_spectrum(ns_sat, d=delta_z, lambdas=lambdas, plot=False)
r_cut, _ = propagation_arbitrary_layers_Born_spectrum(ns_cut, d=delta_z, lambdas=lambdas, plot=False)

plt.figure()
plt.plot(lambdas, r, label="below saturation")
plt.plot(lambdas, r_sat, label="saturation at " + str(saturation*100) + "%")
plt.plot(lambdas, r_cut, label="saturation + cut")
plt.title('Reflected spectrum (including saturation)')
plt.legend()
plt.show()
