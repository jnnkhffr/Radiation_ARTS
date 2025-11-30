"""
Spectral shortwave flux through the atmosphere (Kaysers, Shortwave)
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
# Wellenzahlen 10.000–25.000 cm^-1 ~ sichtbares/UV
kays = np.linspace(10000, 25000, 5000)
freqs = pyarts.arts.convert.kaycm2freq(kays)

# --- Simulation ---
flux, alts = fop(freqs, atm)

# --- Frequenzen aus flux holen ---
if hasattr(flux, "frequencies"):
    freqs_used = flux.frequencies
elif hasattr(flux, "freq"):
    freqs_used = flux.freq
else:
    raise RuntimeError("Keine Frequenzen im flux-Objekt gefunden")

# Frequenzen zurück in Kaysers (cm^-1) umrechnen
kays_used = pyarts.arts.convert.freq2kaycm(freqs_used)

# --- Gesamtflüsse berechnen ---
F_toa = np.trapz(flux.down[0, :], kays_used)
F_surface = np.trapz(flux.down[-1, :], kays_used)
absorbed = F_toa - F_surface

print(f"TOA shortwave flux: {F_toa:.2f} W/m^2")
print(f"Surface shortwave flux: {F_surface:.2f} W/m^2")
print(f"Absorbed in atmosphere: {absorbed:.2f} W/m^2")

# --- Plot mit Kaysers auf der x-Achse ---
plt.figure(figsize=(7,5))
plt.plot(kays_used, flux.down[0, :], label="TOA downwelling flux")
plt.plot(kays_used, flux.down[-1, :], label="Surface downwelling flux")
plt.xlabel("Wavenumber [cm$^{-1}$]")   # jetzt korrekt in Kaysers
plt.ylabel("Spectral flux [W/m$^2$/cm$^{-1}$]")
plt.title("Spectral shortwave flux at TOA and surface")
plt.legend()
plt.grid(True)
plt.tight_layout()
#plt.savefig("shortwave_flux_kayser.png")

plt.savefig("C:/Users/janni/Desktop/shortwave_flux_kayser.png")


