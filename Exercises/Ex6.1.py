# Ex6.1.py — Aufgabe 1: OLR-Spektrum für 1×CO2 und 2×CO2

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pyarts3 as pa

from Ex4_climate_model import climate_column

pa.data.download()


# Atmosphärenprofil erzeugen

Ts = 290.0
Tcp = 200.0
RH = 0.8

p, T, x = climate_column(Ts=Ts, Tcp=Tcp, RH=RH, N=100)
p_grid  = p.astype(float)
T_field = T.astype(float)
H2O_vmr = x.astype(float)


# Höhe aus Hypsometrie berechnen

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


# Atmosphären-Dataset für ARTS

kayser_grid = np.linspace(1, 2000, 300)

atm = xr.Dataset(
    {
        "t":   (("lat","lon","alt"), T_field.reshape(1,1,-1)),
        "p":   (("lat","lon","alt"), p_grid.reshape(1,1,-1)),
        "H2O": (("lat","lon","alt"), H2O_vmr.reshape(1,1,-1)),
        "CO2": (("lat","lon","alt"), np.ones((1,1,len(p_grid))) * 4e-4),
        "O3":  (("lat","lon","alt"), np.ones((1,1,len(p_grid))) * 1e-6),
    },
    coords={
        "lat": [0.0],
        "lon": [0.0],
        "alt": z_field
    }
)


# OLR-Berechnung (robust)

def compute_OLR_spectrum_ARTS(atm_dataset, CO2_vmr):

    ws = pa.workspace.Workspace()

    # Frequenzgitter
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
    ws.propagation_matrix_agendaAuto()

    # Atmosphärenprofil setzen
    atm_mod = atm_dataset.copy()
    atm_mod["CO2"][:] = CO2_vmr
    ws.atmospheric_field = pa.data.to_atmospheric_field(atm_mod)

    # Oberfläche
    ws.surface_fieldPlanet(option="Earth")
    ws.surface_field[pa.arts.SurfaceKey("t")] = float(atm_mod["t"].values[0,0,0])

    # Geometrie
    #z_top = float(atm_mod["alt"].values[-1])
    pos = [100e3, 0.0, 0.0]
    los = [180.0, 0.0]
    ws.ray_pathGeometric(pos=pos, los=los, max_step=2000.0)

    # Radiance
    ws.spectral_radianceClearskyEmission()

    rad = np.array(ws.spectral_radiance.value)
    while rad.ndim > 1:
        rad = rad[:, 0]

    return rad


def integrate_OLR(kayser_grid, spectrum):
    return np.trapezoid(spectrum, kayser_grid)


# 5. Aufgabe 1: Spektren für 400 ppm und 800 ppm

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
#plt.show()

plt.figure(figsize=(10,5))
plt.plot(kayser_grid, OLR_2x - OLR_1x)
plt.xlabel("Wavenumber [cm⁻¹]")
plt.ylabel("ΔOLR (2× - 1×)")
plt.title("Spectral OLR difference (CO₂ doubling)")
plt.grid(True)
#plt.show()


# Aufgabe 2: Radiative Forcing für mehrere CO₂-Faktoren

CO2_factors = [0.25, 0.5, 1, 2, 4, 8]
CO2_base = 4e-4

OLR_values = []
for f in CO2_factors:
    CO2_vmr = CO2_base * f
    spectrum = compute_OLR_spectrum_ARTS(atm, CO2_vmr)
    OLR = integrate_OLR(kayser_grid, spectrum)
    OLR_values.append(OLR)

OLR_1x_int = OLR_values[CO2_factors.index(1)]
forcing = [OLR_1x_int - val for val in OLR_values]

plt.figure(figsize=(8,5))
plt.plot(CO2_factors, forcing, marker="o")
plt.xscale("log", base=2)
plt.xlabel("CO₂ factor (relative to 1×)") # Hier noch den FAktor anpassen
plt.ylabel("Radiative forcing [W/m²]")
plt.title("Radiative forcing for multiple CO₂ doublings")
plt.grid(True)
#plt.show()


# Aufgabe 3: CO2-Forcing als Funktion der Oberflächentemperatur

Ts_values = np.linspace(250, 310, 7)   # 250, 260, ..., 310 K
forcing_Ts = []

for Ts in Ts_values:
    # Neues Klimaprofil erzeugen
    p, T, x = climate_column(Ts=Ts, Tcp=200.0, RH=0.8, N=100)

    # Höhe neu berechnen
    e = x * p
    r = epsilon * e / (p - e)
    q = r / (1.0 + r)
    Tv = T * (1.0 + (Rv / Rd - 1.0) * q)

    z = np.zeros_like(p)
    for i in range(len(p)-1):
        Tv_bar = 0.5 * (Tv[i] + Tv[i+1])
        z[i+1] = z[i] + (Rd/g) * Tv_bar * np.log(p[i]/p[i+1])

    # Atmosphären-Dataset neu bauen
    atm_T = xr.Dataset(
        {
            "t":   (("lat","lon","alt"), T.reshape(1,1,-1)),
            "p":   (("lat","lon","alt"), p.reshape(1,1,-1)),
            "H2O": (("lat","lon","alt"), x.reshape(1,1,-1)),
            "CO2": (("lat","lon","alt"), np.ones((1,1,len(p))) * 4e-4),
            "O3":  (("lat","lon","alt"), np.ones((1,1,len(p))) * 1e-6),
        },
        coords={"lat":[0.0], "lon":[0.0], "alt":z}
    )

    # OLR für 1x und 2x CO2
    OLR_1x_T = integrate_OLR(kayser_grid, compute_OLR_spectrum_ARTS(atm_T, 4e-4))
    OLR_2x_T = integrate_OLR(kayser_grid, compute_OLR_spectrum_ARTS(atm_T, 8e-4))

    forcing_Ts.append(OLR_1x_T - OLR_2x_T)

# Plot
plt.figure(figsize=(8,5))
plt.plot(Ts_values, forcing_Ts, marker="o")
plt.xlabel("Surface temperature Ts [K]")
plt.ylabel("CO₂ forcing ΔF [W/m²]")
plt.title("CO₂ radiative forcing vs. surface temperature")
plt.grid(True)
plt.show()

