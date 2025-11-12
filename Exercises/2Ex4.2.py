"""
*** This is the skript which is working***

Clear-sky outgoing longwave radiation using ARTS,
with custom single-column atmosphere from climate_column().
"""

import os
import numpy as np
#import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyarts3 as pa
import xarray as xr

from Ex4_climate_model import climate_column

# --- Katalogdaten sicherstellen ---
pa.data.download()

# --- Konstanten für dein Säulenmodell ---
Ts = 290.0   # Oberflächentemperatur [K]
Tcp = 200.0  # Tropopausentemperatur [K]
RH = 0.8     # relative Feuchte

# --- Säulenprofil erzeugen ---
p, T, x = climate_column(Ts=Ts, Tcp=Tcp, RH=RH, N=100)
p_grid  = p.astype(float)      # Pa
T_field = T.astype(float)      # K
H2O_vmr = x.astype(float)      # VMR

# --- Höhenfeld via Hypsometrie ---
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

print("p_grid:", p_grid.shape)
print("T_field:", T_field.shape)
print("H2O_vmr:", H2O_vmr.shape)
print("z_field:", z_field.shape)
print("p_grid decreasing:", np.all(np.diff(p_grid) < 0))
print("z_field increasing:", np.all(np.diff(z_field) > 0))
print("z_field min/max:", z_field.min(), z_field.max())

# --- Workspace ---
ws = pa.workspace.Workspace()

# Frequenzgitter (Kayser -> Hz)
kayser_grid = np.linspace(1, 2000, 100)  # cm^-1
ws.frequency_grid = pa.arts.convert.kaycm2freq(kayser_grid)

# Absorptionsspezies (empfohlen vollständige Nomenklatur)
ws.absorption_speciesSet(species=[
    "H2O-161",
    "H2O-ForeignContCKDMT400",
    "H2O-SelfContCKDMT400",
    #"CO2",
    #"O3"
])

# Katalogdaten laden
ws.ReadCatalogData()

# --- Atmosphärenfelder direkt per xarray -> pa.data.to_atmospheric_field ---
# Wir erstellen lat/lon als 1-element Arrays und formen die Feldvariablen auf (lat, lon, alt)

lat_vals = np.array([0.0])   # scalar latitude
lon_vals = np.array([0.0])   # scalar longitude

# sicherstellen: alt (z_field) ist aufsteigend; falls nicht, sortieren wir alle Arrays entsprechend
if not np.all(np.diff(z_field) > 0):
    order = np.argsort(z_field)
    z_field = z_field[order]
    p_grid = p_grid[order]
    T_field = T_field[order]
    H2O_vmr = H2O_vmr[order]

# reshape zu (lat, lon, alt)
t_3d   = T_field.reshape(1, 1, -1)
p_3d   = p_grid.reshape(1, 1, -1)
h2o_3d = H2O_vmr.reshape(1, 1, -1)
alt_1d = z_field             # alt bleibt 1d

atm = xr.Dataset(
    {
        "t": (("lat", "lon", "alt"), t_3d),
        "p": (("lat", "lon", "alt"), p_3d),
        "H2O": (("lat", "lon", "alt"), h2o_3d),
    },
    coords={
        "lat": lat_vals,
        "lon": lon_vals,
        "alt": alt_1d
    }
)

# Attribute (wichtig für to_atmospheric_field)
atm["t"].attrs = {"units": "K", "long_name": "Temperature"}
atm["p"].attrs = {"units": "Pa", "long_name": "Pressure"}
atm["H2O"].attrs = {"units": "mol/mol", "long_name": "Water vapor volume mixing ratio"}
atm["alt"].attrs = {"units": "m", "long_name": "Geometric altitude"}
atm["lat"].attrs = {"units": "degrees_north"}
atm["lon"].attrs = {"units": "degrees_east"}

# Konvertieren und an workspace zuweisen
ws.atmospheric_field = pa.data.to_atmospheric_field(atm)


# --- Quick checks (sicher, ohne nicht vorhandene Workspace-Attribute) ---
print("xarray atm summary:")
print(atm)
print("atm coords:", list(atm.coords))
print("atm dims and shapes:")
for v in ["t", "p", "H2O"]:
    print(f"  {v}: dims={atm[v].dims}, shape={atm[v].shape}")

# Zeige das in den Workspace übertragene Objekt an (repr oder dir sind robust)
print("ws.atmospheric_field (repr):", repr(ws.atmospheric_field))
try:
    print("dir(ws.atmospheric_field):", sorted([k for k in dir(ws.atmospheric_field) if not k.startswith("_")])[:50])
except Exception:
    pass

# Wenn du weiterhin Informationen über das Druckgitter etc. brauchst, prüfe das xarray direkt:
print("pressure top/bottom:", float(atm["p"].isel(alt=0).values), float(atm["p"].isel(alt=-1).values))


# Oberfläche
ws.surface_fieldPlanet(option="Earth")
# Oberfläche explizit aus Profil setzen (Index 0 = Boden)
ws.surface_field[pa.arts.SurfaceKey("t")] = float(T_field[0])

# Propagation agenda automatisch (falls nicht schon gesetzt)
ws.propagation_matrix_agendaAuto()

# Geometrie: Start knapp über Boden
pos = [float(z_field[0] + 1.0), 0.0, 0.0]
los = [180.0, 0.0]
ws.ray_pathGeometric(pos=pos, los=los, max_step=1000.0)

# Strahlung berechnen
ws.spectral_radianceClearskyEmission()

# Plot
fig, ax = plt.subplots()
ax.plot(kayser_grid, ws.spectral_radiance[:, 0], lw=2)
ax.set_xlabel("Frequency / Kayser (cm$^{-1}$)")
ax.set_ylabel("Spectral radiance")
ax.set_title("Clear sky outgoing radiance (climate_column atmosphere)")

if "ARTS_HEADLESS" not in os.environ:
    plt.show()
else:
    #fig.savefig("olr_spectrum.png", dpi=200)
    None
