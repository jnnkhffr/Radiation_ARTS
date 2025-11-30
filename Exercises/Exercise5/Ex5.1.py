import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import h, c, k

T = 5800  # Sun effective temperature
wn = np.linspace(0, 30000, 1000)  # wavenumber range [cm^-1]
nu = wn * 100 * c  # convert cm^-1 to Hz

B_nu = (2*h*nu**3/c**2) / (np.exp(h*nu/(k*T)) - 1)

plt.plot(wn, B_nu)
plt.xlabel("Wavenumber [cm$^{-1}$]")
plt.ylabel("Planck function B$_\\nu$ [W sr$^{-1}$ m$^{-2}$ Hz$^{-1}$]")
plt.title("Planck function at T = 5800 K")
plt.xlim(0, 30000)
plt.show()

