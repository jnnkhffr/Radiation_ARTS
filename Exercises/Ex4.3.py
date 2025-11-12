import os
import numpy as np
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

# Hypsometry for altitude
g = 9.80665; Rd = 287.05; Rv = 461.5; epsilon = Rd / Rv
e = H2O_vmr * p_grid
r = epsilon * e / (p_grid - e)
q = r / (1.0 + r)
Tv = T_field * (1.0 + (Rv / Rd - 1.0) * q)
z_field = np.zeros_like(p_grid)
for i in range(len(p_grid) - 1):
    Tv_bar = 0.5 * (Tv[i] + Tv[i+1])
    z_field[i+1] = z_field[i] + (Rd/g) * Tv_bar * np.log(p_grid[i]/p_grid[i+1])

# Frequency grid
kayser_grid = np.linspace(1, 2000, 1000)  # cm^-1
freq_grid = pa.arts.convert.kaycm2freq(kayser_grid)

# Build base atmosphere (H2O + CO2, no O3 yet)
lat_vals = np.array([0.0]); lon_vals = np.array([0.0])
t_3d   = T_field.reshape(1,1,-1)
p_3d   = p_grid.reshape(1,1,-1)
h2o_3d = H2O_vmr.reshape(1,1,-1)
alt_1d = z_field

atm_base = xr.Dataset(
    {
        "t": (("lat","lon","alt"), t_3d),
        "p": (("lat","lon","alt"), p_3d),
        "H2O": (("lat","lon","alt"), h2o_3d),
        "CO2": (("lat","lon","alt"), np.ones_like(p_3d)*4e-4), # 400 ppm
    },
    coords={"lat":lat_vals,"lon":lon_vals,"alt":alt_1d}
)
atm_base["t"].attrs = {"units":"K"}
atm_base["p"].attrs = {"units":"Pa"}
atm_base["H2O"].attrs = {"units":"mol/mol"}
atm_base["CO2"].attrs = {"units":"mol/mol"}
atm_base["alt"].attrs = {"units":"m"}

# Atmosphere with O3 added (well-mixed 1e-6)
atm_o3 = atm_base.copy(deep=True)
atm_o3["O3"] = (("lat","lon","alt"), np.ones_like(p_3d)*1e-6)
atm_o3["O3"].attrs = {"units":"mol/mol"}

# Function to compute clearsky radiance
def compute_radiance(atm, include_o3=False):
    ws = pa.workspace.Workspace()
    ws.frequency_grid = freq_grid.copy()
    species = ["H2O-161","H2O-ForeignContCKDMT400","H2O-SelfContCKDMT400","CO2-626"]
    if include_o3:
        species.append("O3")
    ws.absorption_speciesSet(species=species)
    ws.ReadCatalogData()
    cutoff = pa.arts.convert.kaycm2freq(25)
    for band in ws.absorption_bands:
        ws.absorption_bands[band].cutoff="ByLine"
        ws.absorption_bands[band].cutoff_value=cutoff
    ws.propagation_matrix_agendaAuto()
    ws.atmospheric_field = pa.data.to_atmospheric_field(atm)
    ws.surface_fieldPlanet(option="Earth")
    ws.surface_field[pa.arts.SurfaceKey("t")] = float(atm["t"].isel(alt=0).item())
    pos=[100e3,0.0,0.0]; los=[180.0,0.0]
    ws.ray_pathGeometric(pos=pos,los=los,max_step=1000.0)
    ws.spectral_radianceClearskyEmission()
    return ws.spectral_radiance[:,0].copy()

# Compute spectra
spec_no_o3 = compute_radiance(atm_base, include_o3=False)
spec_with_o3 = compute_radiance(atm_o3, include_o3=True)

# Convert frequency to wavenumber for plotting
kayser = pa.arts.convert.freq2kaycm(freq_grid)

# Plot
fig, ax = plt.subplots(figsize=(8,5))
ax.plot(kayser, spec_no_o3, label="No O3", lw=2)
ax.plot(kayser, spec_with_o3, label="With O3 (1e-6)", lw=2, linestyle="--")
ax.set_xlabel("Wavenumber (cm$^{-1}$)")
ax.set_ylabel("Spectral radiance")
ax.set_title("Clear-sky outgoing radiance: effect of O3")
ax.legend(); ax.grid(True)
if "ARTS_HEADLESS" not in os.environ:
    plt.show()

# Integrate OLR
def integrate_olr(freq, spec):
    rad = np.trapz(spec, x=freq)   # integrate over Hz
    flux = np.pi * rad             # multiply by pi for flux
    return flux

olr_no_o3 = integrate_olr(freq_grid, spec_no_o3)
olr_with_o3 = integrate_olr(freq_grid, spec_with_o3)
delta = olr_with_o3 - olr_no_o3
pct = 100.0*delta/olr_no_o3

print(f"OLR without O3: {olr_no_o3:.3f} W/m^2")
print(f"OLR with O3 (1e-6): {olr_with_o3:.3f} W/m^2")
print(f"Absolute change: {delta:.3f} W/m^2")
print(f"Relative change: {pct:.3f} %")


