# Ex6.1.py — Aufgabe 1: OLR-Spektrum für 1×CO2 und 2×CO2

import os
import numpy as np
import matplotlib.pyplot as plt
import pyarts3 as pa
import xarray as xr

from Ex4_climate_model import climate_column   # dein Modell

import pyarts3 as pa
pa.data.download()


# ============================================================
# 1. Atmosphärenprofil erzeugen
# ============================================================

Ts = 290.0
Tcp = 200.0
RH = 0.8

p, T, x = climate_column(Ts=Ts, Tcp=Tcp, RH=RH, N=100)
p_grid  = p.astype(float)
T_field = T.astype(float)
H2O_vmr = x.astype(float)

# ============================================================
# 2. Höhe aus Hypsometrie berechnen
# ============================================================

g = 9.80665
Rd = 287.05
Rv = 461.5
epsilon = Rd / Rv

e = H2O_vmr * p_grid
r = epsilon * e / (p_grid - e)
q = r / (1.0 + r)
Tv = T_field * (1.0 + (Rv / Rd - 1.0) * q)

z_field = np.zeros_like(p_grid)
for i in range(len(p_grid) - 1):
    Tv_bar = 0.5 * (Tv[i] + Tv[i+1])
    z_field[i+1] = z_field[i] + (Rd / g) * Tv_bar * np.log(p_grid[i] / p_grid[i+1])

# ============================================================
# 3. ARTS Workspace vorbereiten
# ============================================================

ws = pa.workspace.Workspace()

# Frequenzgitter (Kayser → Hz)
kayser_grid = np.linspace(1, 2000, 300)
ws.frequency_grid = pa.arts.convert.kaycm2freq(kayser_grid)

# Absorptionsspezies
ws.absorption_speciesSet(species=[
    "H2O-161",
    "H2O-ForeignContCKDMT400",
    "H2O-SelfContCKDMT400",
    "CO2-626",
    "O3"
])

ws.ReadCatalogData()

cutoff = pa.arts.convert.kaycm2freq(25)
for band in ws.absorption_bands:
    ws.absorption_bands[band].cutoff = "ByLine"
    ws.absorption_bands[band].cutoff_value = cutoff

ws.absorption_bands.keep_hitran_s(approximate_percentile=90)
ws.propagation_matrix_agendaAuto()

# ============================================================
# 4. Atmosphären-Dataset für ARTS bauen
# ============================================================

atm = xr.Dataset(
    {
        "t":   (("lat","lon","alt"), T_field.reshape(1,1,-1)),
        "p":   (("lat","lon","alt"), p_grid.reshape(1,1,-1)),
        "H2O": (("lat","lon","alt"), H2O_vmr.reshape(1,1,-1)),
        "CO2": (("lat","lon","alt"), np.ones((1,1,len(p_grid))) * 4e-4),  # 400 ppm
        "O3":  (("lat","lon","alt"), np.ones((1,1,len(p_grid))) * 1e-6),
    },
    coords={
        "lat": [0.0],
        "lon": [0.0],
        "alt": z_field
    }
)

ws.atmospheric_field = pa.data.to_atmospheric_field(atm)

# Oberfläche setzen
ws.surface_fieldPlanet(option="Earth")
ws.surface_field[pa.arts.SurfaceKey("t")] = float(T_field[0])

# Strahlungsgeometrie
pos = [100e3, 0.0, 0.0]
los = [180.0, 0.0]
ws.ray_pathGeometric(pos=pos, los=los, max_step=1000.0)

# ============================================================
# 5. Funktionen für OLR-Berechnung
# ============================================================

def compute_OLR_spectrum_ARTS(atm_dataset, CO2_vmr):

    ws = pa.workspace.Workspace()

    ws.frequency_grid = pa.arts.convert.kaycm2freq(kayser_grid)

    ws.absorption_speciesSet(species=[
        "H2O-161",
        "H2O-ForeignContCKDMT400",
        "H2O-SelfContCKDMT400",
        "CO2-626",
        "O3"
    ])
    ws.ReadCatalogData()
    ws.propagation_matrix_agendaAuto()

    atm_mod = atm_dataset.copy()
    atm_mod["CO2"][:] = CO2_vmr

    ws.atmospheric_field = pa.data.to_atmospheric_field(atm_mod)

    ws.surface_fieldPlanet(option="Earth")
    ws.surface_field[pa.arts.SurfaceKey("t")] = float(atm_mod["t"].values[0,0,0])

    # Beobachter knapp über TOA
    z_top = float(atm_mod["alt"].values[-1])
    pos = [100e3, 0.0, 0.0]
    los = [180.0, 0.0]

    ws.ray_pathGeometric(pos=pos, los=los, max_step=2000.0)

    ws.spectral_radianceClearskyEmission()

    rad = np.array(ws.spectral_radiance.value)
    while rad.ndim > 1:
        rad = rad[:, 0]

    return rad


def integrate_OLR(kayser_grid, spectrum):
    return np.trapezoid(spectrum, kayser_grid)

# ============================================================
# 6. Aufgabe 1: CO₂ verdoppeln & Spektren vergleichen
# ============================================================

OLR_1x = compute_OLR_spectrum_ARTS(atm, 4e-4)
OLR_2x = compute_OLR_spectrum_ARTS(atm, 8e-4)

plt.figure(figsize=(10,5))
plt.plot(kayser_grid, OLR_1x, label="400 ppm CO₂")
plt.plot(kayser_grid, OLR_2x, label="800 ppm CO₂")
plt.xlabel("Wavenumber [cm⁻¹]")
plt.ylabel("Spectral radiance")
plt.title("OLR spectrum for 1× and 2× CO₂")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10,5))
plt.plot(kayser_grid, OLR_2x - OLR_1x)
plt.xlabel("Wavenumber [cm⁻¹]")
plt.ylabel("ΔOLR (2× - 1×)")
plt.title("Spectral OLR difference (CO₂ doubling)")
plt.grid(True)
plt.show()




#Nr.6.2
CO2_factors = [0.25, 0.5, 1, 2, 4, 8]
CO2_base = 4e-4

OLR_values = []
for f in CO2_factors:
    CO2_vmr = CO2_base * f
    spectrum = compute_OLR_spectrum_ARTS(atm, CO2_vmr)
    OLR = integrate_OLR(kayser_grid, spectrum)
    OLR_values.append(OLR)

OLR_1x = OLR_values[CO2_factors.index(1)]
forcing = [OLR_1x - val for val in OLR_values]

plt.figure(figsize=(8,5))
plt.plot(CO2_factors, forcing, marker="o")
plt.xscale("log", base=2)
plt.xlabel("CO₂ factor (relative to 1×)")
plt.ylabel("Radiative forcing [W/m²]")
plt.title("Radiative forcing for multiple CO₂ doublings")
plt.grid(True)
plt.show()
