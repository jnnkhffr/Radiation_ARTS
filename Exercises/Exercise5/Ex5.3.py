"""
Shortwave atmospheric absorption vs. surface albedo (Kaysers, robust)
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
import pyarts3 as pyarts

# --- Daten sicherstellen ---
pyarts.data.download()  # lädt Standardkataloge inkl. Sonne

# --- Geometrie: Sonne sicher über dem Horizont, co-located mit Atmosphäre ---
atm_lat = 0.0
atm_lon = 0.0
sol_lat = 20.0     # stabile, nicht-grenzwertige Sonnenhöhe
sol_lon = 0.0

# --- Frequenzbereich: breiter SW, danach auf tatsächliche Länge kürzen ---
kays_in = np.linspace(1000, 30000, 4000)                 # cm^-1
freqs_in = pyarts.arts.convert.kaycm2freq(kays_in)       # Hz

# --- Albedo-Werte ---
albedos = np.linspace(0.0, 1.0, 11)  # 0.0, 0.1, ..., 1.0
abs_sw_list = []

for alb in albedos:
    # Operator mit aktuellem Albedo-Wert
    fop = pyarts.recipe.SpectralAtmosphericFlux(
        species=["H2O-161", "O2-66", "N2-44", "CO2-626", "O3-XFIT"],
        remove_lines_percentile={"H2O": 70},
        atmospheric_altitude=50e3,
        visible_surface_reflectivity=alb,
        atm_latitude=atm_lat,
        atm_longitude=atm_lon,
        solar_latitude=sol_lat,
        solar_longitude=sol_lon,
    )

    atm = fop.get_atmosphere()

    # Simulation
    flux, alts = fop(freqs_in, atm)

    # Achsenkonsistenz
    nfreq = flux.up.shape[0]  # Arrays sind [n_freq, n_alt]
    kays_used = pyarts.arts.convert.freq2kaycm(freqs_in[:nfreq])

    # Indizes Oberfläche/TOA
    i_sfc = int(np.argmin(alts))
    i_toa = int(np.argmax(alts))

    # Spektrale Flüsse
    down_toa_spec = flux.direct_down[:, i_toa] + flux.diffuse_down[:, i_toa]
    down_sfc_spec = flux.direct_down[:, i_sfc] + flux.diffuse_down[:, i_sfc]
    up_toa_spec   = flux.up[:, i_toa]
    up_sfc_spec   = flux.up[:, i_sfc]

    # Nettofluss (down - up) auf TOA und an der Oberfläche
    net_toa_spec = down_toa_spec - up_toa_spec
    net_sfc_spec = down_sfc_spec - up_sfc_spec

    # Integration über Kaysers: atmosphärische Absorption = Abnahme des Nettoflusses
    Net_TOA = np.trapezoid(net_toa_spec, kays_used)      # W/m^2
    Net_SFC = np.trapezoid(net_sfc_spec, kays_used)      # W/m^2
    absorbed_sw = Net_TOA - Net_SFC                      # W/m^2

    abs_sw_list.append(absorbed_sw)
    print(f"Albedo {alb:.2f} -> Atmospheric SW absorption: {absorbed_sw:.2f} W/m^2")

# --- Plot: Absorption vs. Albedo ---
plt.figure(figsize=(7.5,5))
plt.plot(albedos, abs_sw_list, marker="o")
plt.xlabel("Visible surface reflectivity (albedo)")
plt.ylabel("Atmospheric shortwave absorption [W/m$^2$]")
plt.title("Atmospheric shortwave absorption vs. surface albedo")
plt.grid(True, alpha=0.35)
plt.tight_layout()
plt.savefig("C:/Users/janni/Desktop/sw_absorption_vs_albedo.png")
