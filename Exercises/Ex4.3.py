"""
*** This is the skript which is working***

Clear-sky outgoing longwave radiation using ARTS,
with custom single-column atmosphere from climate_column() with Ozon.
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
kayser_grid = np.linspace(1, 2000, 1000)  # cm^-1
ws.frequency_grid = pa.arts.convert.kaycm2freq(kayser_grid)

# Absorption species (recommended full nomenclature)
ws.absorption_speciesSet(species=[
    "H2O-161",
    "H2O-ForeignContCKDMT400",
    "H2O-SelfContCKDMT400",
    # "CO2-626",
    # "O3"
])

# Load catalog data
ws.ReadCatalogData()

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
    },
    coords={
        "lat": lat_vals,
        "lon": lon_vals,
        "alt": alt_1d
    }
)

# Attributes (important for to_atmospheric_field)
atm["t"].attrs = {"units": "K", "long_name": "Temperature"}
atm["p"].attrs = {"units": "Pa", "long_name": "Pressure"}
atm["H2O"].attrs = {"units": "mol/mol", "long_name": "Water vapor volume mixing ratio"}
atm["alt"].attrs = {"units": "m", "long_name": "Geometric altitude"}
atm["lat"].attrs = {"units": "degrees_north"}
atm["lon"].attrs = {"units": "degrees_east"}

# Convert and assign to workspace
ws.atmospheric_field = pa.data.to_atmospheric_field(atm)

# Add ozone as well-mixed gas and compute/plot spectra with and without O3

def run_clearsky_for_atm(xatm, include_o3=False, o3_vmr=1e-6, freq_grid=None):
    ws_loc = pa.workspace.Workspace()
    if freq_grid is not None:
        ws_loc.frequency_grid = freq_grid.copy()
    else:
        ws_loc.frequency_grid = ws.frequency_grid.copy()

    # prepare species list (include O3 if requested)
    species_list = [
        "H2O-161",
        "H2O-ForeignContCKDMT400",
        "H2O-SelfContCKDMT400",
    ]
    if include_o3:
        species_list.append("O3")
    ws_loc.absorption_speciesSet(species=species_list)
    ws_loc.ReadCatalogData()

    # copy and add O3 if requested (ensure this is done BEFORE to_atmospheric_field)
    atm_use = xatm.copy(deep=True)
    if include_o3:
        if "O3" not in atm_use:
            lat_len = atm_use.sizes["lat"]
            lon_len = atm_use.sizes["lon"]
            alt_len = atm_use.sizes["alt"]
            atm_use["O3"] = (("lat", "lon", "alt"),
                             np.full((lat_len, lon_len, alt_len), o3_vmr, dtype=float))
            atm_use["O3"].attrs = {"units": "mol/mol", "long_name": "Ozone volume mixing ratio"}

    # debug: confirm O3 present in atm_use
    print("include_o3:", include_o3, "O3 in atm_use:", "O3" in atm_use)
    if "O3" in atm_use:
        print("O3 min/max:", float(atm_use["O3"].min()), float(atm_use["O3"].max()))

    # assign atmospheric field (now with O3 if requested)
    ws_loc.atmospheric_field = pa.data.to_atmospheric_field(atm_use)

    # debug: check species known in atmospheric_field
    try:
        print("species_keys in atmospheric_field:", ws_loc.atmospheric_field.species_keys())
    except Exception as e:
        print("could not query species_keys:", e)

    # continue setup
    ws_loc.surface_fieldPlanet(option="Earth")
    ws_loc.surface_field[pa.arts.SurfaceKey("t")] = float(atm_use["t"].isel(lat=0, lon=0, alt=0).item())
    ws_loc.propagation_matrix_agendaAuto()
    pos_local = [float(atm_use["alt"].values[0] + 1.0), 0.0, 0.0]
    ws_loc.ray_pathGeometric(pos=pos_local, los=[180.0, 0.0], max_step=1000.0)
    ws_loc.spectral_radianceClearskyEmission()

    freq = ws_loc.frequency_grid.copy()
    spec = ws_loc.spectral_radiance[:, 0].copy()
    return ws_loc, freq, spec


# run for atmosphere without O3
ws_no_o3, freq_no_o3, spec_no_o3 = run_clearsky_for_atm(atm, include_o3=False)

# run for atmosphere with well-mixed O3 (VMR = 1e-6)
ws_o3, freq_o3, spec_o3 = run_clearsky_for_atm(atm, include_o3=True, o3_vmr=1e-6)

# convert frequency to wavenumber (cm^-1) for plotting on x-axis
kayser = pa.arts.convert.freq2kaycm(freq_no_o3)

# plot both spectra
fig, ax = plt.subplots(figsize=(8,5))
ax.plot(kayser, spec_no_o3, label="No O3", lw=2)
ax.plot(kayser, spec_o3, label="With O3 (1e-6)", lw=2, linestyle="--")
ax.set_xlabel("Wavenumber (cm$^{-1}$)")
ax.set_ylabel("Spectral radiance (W m$^{-2}$ sr$^{-1}$ Hz$^{-1}$)")
ax.set_title("Clear-sky spectral radiance: without and with well-mixed O3 (1e-6)")
ax.legend()
ax.grid(True)

if "ARTS_HEADLESS" not in os.environ:
    plt.show()
else:
    fig.savefig("spectra_o3_comparison.png", dpi=200)

# (Optional) integrate to get OLR in W/m^2 for both cases
def integrate_olr(freq, spec):
    # integrate radiance over frequency (Hz) and multiply by pi to get flux
    rad = np.trapz(spec, x=freq)   # W m^-2 sr^-1
    flux = np.pi * rad             # W m^-2
    return flux

olr_no_o3 = integrate_olr(freq_no_o3, spec_no_o3)
olr_with_o3 = integrate_olr(freq_o3, spec_o3)
delta = olr_with_o3 - olr_no_o3
pct = 100.0 * delta / olr_no_o3 if olr_no_o3 != 0 else np.nan

print(f"OLR without O3: {olr_no_o3:.3f} W/m^2")
print(f"OLR with O3 (1e-6): {olr_with_o3:.3f} W/m^2")
print(f"Absolute change: {delta:.3f} W/m^2")
print(f"Relative change: {pct:.3f} %")
