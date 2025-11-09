"""
Clear-sky outgoing longwave radiation using ARTS,
with custom single-column atmosphere from climate_column().
"""

import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyarts3 as pa

# Importiere dein Atmosphärenmodell
from Ex4_climate_model import climate_column

# Katalogdaten sicherstellen
pa.data.download()

# Konstanten für dein Säulenmodell
Ts = 290.0   # Oberflächentemperatur [K]
Tcp = 200.0  # Tropopausentemperatur [K]
RH = 0.8     # relative Feuchte

# Säulenprofil erzeugen
p, T, x = climate_column(Ts=Ts, Tcp=Tcp, RH=RH, N=100)
p_grid = np.flip(p).astype(float)      # Pa, aufsteigend
T_field = np.flip(T).astype(float)     # K
H2O_vmr = np.flip(x).astype(float)     # VMR

# Höhenfeld via Hypsometrie
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

# Workspace
ws = pa.workspace.Workspace()

# Frequenzgitter
kayser_grid = np.linspace(1, 2000, 200)  # cm^-1
ws.frequency_grid = pa.arts.convert.kaycm2freq(kayser_grid)


# Absorptionsspezies
ws.absorption_speciesSet(
    species=["H2O", "H2O-ForeignContCKDMT400", "H2O-SelfContCKDMT400"]
)

# Katalogdaten laden
ws.ReadCatalogData()

# Cutoff
cutoff = pa.arts.convert.kaycm2freq(25)
for band in ws.absorption_bands:
    ws.absorption_bands[band].cutoff = "ByLine"
    ws.absorption_bands[band].cutoff_value = cutoff

# Linien reduzieren
ws.absorption_bands.keep_hitran_s(approximate_percentile=90)

# Absorptionsagenda
ws.propagation_matrix_agendaAuto()

# Oberfläche
ws.surface_fieldPlanet(option="Earth")
ws.surface_field[pa.arts.SurfaceKey("t")] = Ts

# Atmosphärenfelder setzen mit ARTS-Typen
ws.pressure_grid     = pa.arts.Vector(p_grid)
#Breite/Länge muss NumPy-Array bleiben
ws.latitude_grid     = np.array([0.0])
ws.longitude_grid    = np.array([0.0])

ws.temperature_field = T_field[:, None, None]
ws.z_field           = z_field[:, None, None]
ws.vmr_field         = [H2O_vmr[:, None, None]]

# Geometrie
pos = [z_field[-1], 0.0, 0.0]   # oberste Höhe
los = [180.0, 0.0]
ws.ray_pathGeometric(pos=pos, los=los, max_step=1000.0)

# Strahlung berechnen
ws.spectral_radianceClearskyEmission()

# Plot
fig, ax = plt.subplots()
ax.plot(kayser_grid, ws.spectral_radiance[:, 0])
ax.set_xlabel("Frequency / Kayser (cm$^{-1}$)")
ax.set_ylabel("Spectral radiance")
ax.set_title("Clear sky outgoing radiance (climate_column atmosphere)")

if "ARTS_HEADLESS" not in os.environ:
    plt.show()


