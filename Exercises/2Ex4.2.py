"""
*** This is the skript which is working***

Clear-sky outgoing longwave radiation using ARTS,
with custom single-column atmosphere from climate_column().
"""

import os
import numpy as np
# import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyarts3 as pa
import xarray as xr

from Ex4_climate_model import climate_column

# Ensure catalog data is available
pa.data.download()

# Constants for the column model
Ts = 290.0   # surface temperature [K]
Tcp = 200.0  # tropopause temperature [K]
RH = 0.8     # relative humidity

# Generate column profile
p, T, x = climate_column(Ts=Ts, Tcp=Tcp, RH=RH, N=100)
p_grid  = p.astype(float)      # Pa
T_field = T.astype(float)      # K
H2O_vmr = x.astype(float)      # VMR

# Height field via hypsometry
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

# Create workspace
ws = pa.workspace.Workspace()

# Frequency grid (Kayser -> Hz)
kayser_grid = np.linspace(1, 2000, 300)  # cm^-1
ws.frequency_grid = pa.arts.convert.kaycm2freq(kayser_grid)


# Absorption species (realistic setup)
ws.absorption_speciesSet(species=[
    "H2O-161",
    "H2O-ForeignContCKDMT400",
    "H2O-SelfContCKDMT400",
    "CO2-626",
    "O3"
])

# Load catalog data
ws.ReadCatalogData()

# Apply cutoff for CKD consistency
cutoff = pa.arts.convert.kaycm2freq(25)
for band in ws.absorption_bands:
    ws.absorption_bands[band].cutoff = "ByLine"
    ws.absorption_bands[band].cutoff_value = cutoff

# Optional: remove weak lines to speed up
ws.absorption_bands.keep_hitran_s(approximate_percentile=90)

# Automatic propagation agenda
ws.propagation_matrix_agendaAuto()


# Create lat/lon as 1-element arrays and shape field variables to (lat, lon, alt)
lat_vals = np.array([0.0])   # scalar latitude
lon_vals = np.array([0.0])   # scalar longitude

# Ensure altitude (z_field) is ascending; if not, sort all arrays accordingly
if not np.all(np.diff(z_field) > 0):
    order = np.argsort(z_field)
    z_field = z_field[order]
    p_grid = p_grid[order]
    T_field = T_field[order]
    H2O_vmr = H2O_vmr[order]

# Reshape to (lat, lon, alt)
t_3d   = T_field.reshape(1, 1, -1)
p_3d   = p_grid.reshape(1, 1, -1)
h2o_3d = H2O_vmr.reshape(1, 1, -1)
alt_1d = z_field             # alt remains 1d

atm = xr.Dataset(
    {
        "t": (("lat", "lon", "alt"), t_3d),
        "p": (("lat", "lon", "alt"), p_3d),
        "H2O": (("lat", "lon", "alt"), h2o_3d),
        "CO2": (("lat", "lon", "alt"), np.ones_like(p_3d) * 4e-4),   # 400 ppm
        "O3":  (("lat", "lon", "alt"), np.ones_like(p_3d) * 1e-6),   # 1 ppm
    },
    coords={
        "lat": lat_vals,
        "lon": lon_vals,
        "alt": alt_1d
    }
)

# Attribute ergänzen
atm["CO2"].attrs = {"units": "mol/mol", "long_name": "Carbon dioxide volume mixing ratio"}
atm["O3"].attrs  = {"units": "mol/mol", "long_name": "Ozone volume mixing ratio"}


# Attributes (important for to_atmospheric_field)
atm["t"].attrs = {"units": "K", "long_name": "Temperature"}
atm["p"].attrs = {"units": "Pa", "long_name": "Pressure"}
atm["H2O"].attrs = {"units": "mol/mol", "long_name": "Water vapor volume mixing ratio"}
atm["alt"].attrs = {"units": "m", "long_name": "Geometric altitude"}
atm["lat"].attrs = {"units": "degrees_north"}
atm["lon"].attrs = {"units": "degrees_east"}

# Convert and assign to workspace
ws.atmospheric_field = pa.data.to_atmospheric_field(atm)

# Quick checks (safe, without relying on potentially missing Workspace attributes)
print("xarray atm summary:")
print(atm)
print("atm coords:", list(atm.coords))
print("atm dims and shapes:")
for v in ["t", "p", "H2O"]:
    print(f"  {v}: dims={atm[v].dims}, shape={atm[v].shape}")

# Show the object transferred to the workspace (repr or dir are robust)
print("ws.atmospheric_field (repr):", repr(ws.atmospheric_field))
try:
    print("dir(ws.atmospheric_field):", sorted([k for k in dir(ws.atmospheric_field) if not k.startswith("_")])[:50])
except Exception:
    pass

# If you still need pressure info, check the xarray directly
print("pressure top/bottom:", float(atm["p"].isel(alt=0).values), float(atm["p"].isel(alt=-1).values))

# Surface
ws.surface_fieldPlanet(option="Earth")
# Explicitly set the surface temperature from the profile (index 0 = surface)
ws.surface_field[pa.arts.SurfaceKey("t")] = float(T_field[0])

# Automatic propagation agenda (if not already set)
ws.propagation_matrix_agendaAuto()

# Geometry: start just above the surface
pos = [100e3, 0.0, 0.0]
los = [180.0, 0.0]
ws.ray_pathGeometric(pos=pos, los=los, max_step=1000.0)

# Compute radiance
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
    None
