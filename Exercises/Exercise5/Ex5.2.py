"""
Spectral shortwave flux through the atmosphere (Kaysers, Shortwave, robust)
"""

import matplotlib
matplotlib.use("Agg")  # headless Plot, vermeidet Qt-Probleme

import numpy as np
import matplotlib.pyplot as plt
import pyarts3 as pyarts

# --- Daten laden ---
pyarts.data.download()

# --- Operator initialisieren ---
#fop = pyarts.recipe.SpectralAtmosphericFlux(
#    species=["H2O-161", "O2-66", "N2-44", "CO2-626", "O3-XFIT"],
#    remove_lines_percentile={"H2O": 70},
#    atmospheric_altitude=50e3,      # TOA bei 50 km
#    visible_surface_reflectivity=0.3,
#)
fop = pyarts.recipe.SpectralAtmosphericFlux(
    species=["H2O-161", "O2-66", "N2-44", "CO2-626", "O3-XFIT"],
    remove_lines_percentile={"H2O": 70},
    atmospheric_altitude=50e3,
    visible_surface_reflectivity= 1 #0.3,
    #solar_source="kurucz"   # oder "sun" je nach pyarts-Version
)


# --- Atmosphäre holen (optional anpassbar) ---
atm = fop.get_atmosphere()

# --- Kurzwelliger Bereich (Solar): 10k–25k cm^-1 ---
kays_in = np.linspace(10000, 25000, 5000)                          # Kaysers (cm^-1)
freqs_in = pyarts.arts.convert.kaycm2freq(kays_in)                  # Hz
# --- Kurzwelliger Bereich (Solar): sinnvoller Bereich für ARTS-Sonne ---
#kays_in = np.linspace(400, 2500, 5000)                 # cm^-1
#freqs_in = pyarts.arts.convert.kaycm2freq(kays_in)     # Hz


# --- Simulation ---
flux, alts = fop(freqs_in, atm)
# flux.up, flux.diffuse_down, flux.direct_down haben Form: [n_freq, n_alt]
# flux.down = diffuse + direct

# --- Achsenrobustheit: benutze genau so viele Frequenzpunkte wie flux berechnet hat ---
nfreq = flux.down.shape[0]
freqs_used = freqs_in[:nfreq]                                       # Hz, Länge passt zu flux
kays_used = pyarts.arts.convert.freq2kaycm(freqs_used)              # zurück in cm^-1

# --- TOA und Oberfläche robust bestimmen über Höhe ---
# alts ist mittlere Layerhöhe; wir nehmen Index des Minimums als Oberfläche, Maximum als TOA
i_sfc = int(np.argmin(alts))
i_toa = int(np.argmax(alts))

# Für die Flussberechnung
# (down - up) from TOA - (down -up) from surface


# --- Gesamtflüsse (über Kaysers integrieren) ---
# Spektrale Flüsse sind entlang der Frequenzachse (0) verteilt, Integration über kays_used
#F_toa_down = np.trapz(flux.down[:, i_toa], kays_used)               # W/m^2
#F_sfc_down = np.trapz(flux.down[:, i_sfc], kays_used)               # W/m^2
#absorbed_sw = F_toa_down - F_sfc_down
# --- Gesamtflüsse (über Frequenz integrieren) ---
F_toa_down = np.trapz(flux.down[:, i_toa], freqs_used)              # W/m^2
F_sfc_down = np.trapz(flux.down[:, i_sfc], freqs_used)              # W/m^2
absorbed_sw = F_toa_down - F_sfc_down


print(f"TOA shortwave downwelling flux: {F_toa_down:.2f} W/m^2")
print(f"Surface shortwave downwelling flux: {F_sfc_down:.2f} W/m^2")
print(f"Absorbed in atmosphere (shortwave): {absorbed_sw:.2f} W/m^2")

# --- Plot: Spektralflüsse am TOA und an der Oberfläche (x-Achse in Kaysers) ---
plt.figure(figsize=(8,5))
plt.plot(kays_used, flux.down[:, i_toa], label="TOA downwelling flux")
plt.plot(kays_used, flux.down[:, i_sfc], label="Surface downwelling flux")
plt.xlabel("Wavenumber [cm$^{-1}$]")               # Kaysers
plt.ylabel("Spectral flux [W/m$^2$/cm$^{-1}$]")    # pro Kaysers
plt.title("Spectral shortwave flux at TOA and surface")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
#plt.savefig("shortwave_flux_kayser.png")           # im aktuellen Arbeitsverzeichnis speichern

plt.savefig("C:/Users/janni/Desktop/v2shortwave_flux_kayser.png")


