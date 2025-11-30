"""
Spectral shortwave flux through the atmosphere (gefixt)
"""

import matplotlib
matplotlib.use("Agg")   # verhindert Qt-Fehler in PyCharm

import numpy as np
import matplotlib.pyplot as plt
import pyarts3 as pyarts

# --- Daten laden ---
pyarts.data.download()

# --- Operator initialisieren ---
fop = pyarts.recipe.SpectralAtmosphericFlux(
    species=["H2O-161", "O2-66", "N2-44", "CO2-626", "O3-XFIT"],
    remove_lines_percentile={"H2O": 70},
)

atm = fop.get_atmosphere()

# --- Kurzwelliger Bereich (Solarstrahlung) ---
# Wir geben einen Bereich vor, aber ARTS reduziert intern das Gitter
kays = np.linspace(10000, 25000, 5000)
freqs = pyarts.arts.convert.kaycm2freq(kays)

# --- Simulation ---
flux, alts = fop(freqs, atm)

# --- Frequenzen aus der Rückgabe verwenden ---
# Je nach Version heißt es flux.frequencies oder flux.freq
if hasattr(flux, "frequencies"):
    freqs_used = flux.frequencies
elif hasattr(flux, "freq"):
    freqs_used = flux.freq
else:
    freqs_used = np.arange(flux.down.shape[1])  # fallback

# --- Gesamtflüsse berechnen ---
F_toa = np.trapz(flux.down[0, :], freqs_used)
F_surface = np.trapz(flux.down[-1, :], freqs_used)
absorbed = F_toa - F_surface

print(f"TOA shortwave flux: {F_toa:.2f} W/m^2")
print(f"Surface shortwave flux: {F_surface:.2f} W/m^2")
print(f"Absorbed in atmosphere: {absorbed:.2f} W/m^2")

# --- Plot ---
plt.figure(figsize=(7,5))
plt.plot(freqs_used, flux.down[0, :], label="TOA downwelling flux")
plt.plot(freqs_used, flux.down[-1, :], label="Surface downwelling flux")
plt.xlabel("Frequency [Hz]")  # oder Wavenumber, je nach freqs_used
plt.ylabel("Spectral flux [W/m^2/Hz]")
plt.title("Spectral shortwave flux at TOA and surface")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("C:/Users/janni/Desktop/shortwave_flux.png")
  # Bild speichern statt Qt-Show
#plt.show()
