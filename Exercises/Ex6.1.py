# Ex6.1.py — Aufgabe 1: OLR-Spektrum für 1×CO2 und 2×CO2

import matplotlib
matplotlib.use("Agg")   # verhindert Qt-Fehler in PyCharm

import numpy as np
import matplotlib.pyplot as plt

# Klimamodell importieren (NICHT im Skript einbauen!)
from Ex4_climate_model import climate_column


# ---------------------------------------------------------
#  compute_OLR_spectrum() — vereinfachtes Spektralmodell
# ---------------------------------------------------------

def planck(nu, T):
    h = 6.626e-34
    c = 3e8
    k = 1.381e-23
    nu_SI = nu * 100 * c
    return (2*h*nu_SI**3 / c**2) / (np.exp(h*nu_SI/(k*T)) - 1)

def compute_OLR_spectrum(p, T, x, co2_factor=1.0):

    # Spektralbereich
    nu = np.linspace(400, 2000, 800)

    # CO2 15 µm Band
    co2_center = 667
    co2_width = 180
    k_co2 = 1e-20 * np.exp(-((nu - co2_center)**2) / (2 * co2_width**2))

    # H2O breitband
    k_h2o = 5e-22 * (1 + 0.5*np.exp(-(nu-1500)**2/200**2))

    dp = np.abs(np.gradient(p))
    tau = np.zeros((len(nu), len(p)))

    for i in range(len(p)):
        tau[:, i] = (k_co2 * co2_factor + k_h2o * x[i]) * dp[i]

    tau_cum = np.cumsum(tau[:, ::-1], axis=1)[:, ::-1]

    OLR = np.zeros(len(nu))
    for j in range(len(nu)):
        idx = np.argmin(np.abs(tau_cum[j] - 1))
        OLR[j] = np.pi * planck(nu[j], T[idx])

    return nu, OLR


# ---------------------------------------------------------
#  Aufgabe 6.1: OLR für 1× und 2× CO2
# ---------------------------------------------------------

Ts = 300
Tcp = 200
RH = 0.8

# Klimamodell aufrufen
p, T, x = climate_column(Ts, Tcp, RH)

# OLR berechnen
nu, OLR_1x = compute_OLR_spectrum(p, T, x, co2_factor=1.0)
_,  OLR_2x = compute_OLR_spectrum(p, T, x, co2_factor=2.0)

# Plotten
plt.figure(figsize=(10,5))
plt.plot(nu, OLR_1x, label="1× CO₂")
plt.plot(nu, OLR_2x, label="2× CO₂")
plt.xlabel("Wavenumber [cm⁻¹]")
plt.ylabel("OLR [W/m²/cm⁻¹]")
plt.title("OLR Spectrum for 1× and 2× CO₂")
plt.grid()
plt.legend()
#plt.savefig("OLR_spectrum_CO2_doubling.png")
plt.show

# Radiative Forcing
F_inst = np.trapezoid(OLR_1x - OLR_2x, nu)
print("Instantaneous Radiative Forcing =", F_inst, "W/m²")
