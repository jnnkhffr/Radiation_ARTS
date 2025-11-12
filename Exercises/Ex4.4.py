import os
import matplotlib.pyplot as plt
import numpy as np
import pyarts3 as pa
import scipy.constants as const

# Download ARTS catalogs if they are not already present
pa.data.download()

# ARTS workspace
ws = pa.workspace.Workspace()

# Set up frequency grid
kayser_grid = np.linspace(1, 2000, 2000)  # in Kayser (cm^-1)
ws.frequency_grid = pa.arts.convert.kaycm2freq(kayser_grid)  # in Hz

# Select absorption species and continuum model
ws.absorption_speciesSet(
    species=["H2O-161", "H2O-ForeignContCKDMT400", "H2O-SelfContCKDMT400", "CO2-626", "O3"]
)

# Read spectral line data from ARTS catalog
ws.ReadCatalogData()

# Apply a frequency cutoff (25 cm^-1 for CKD consistency)
cutoff = pa.arts.convert.kaycm2freq(25)
for band in ws.absorption_bands:
    ws.absorption_bands[band].cutoff = "ByLine"
    ws.absorption_bands[band].cutoff_value = cutoff

# Remove 90% of the lines to speed up the calculation
ws.absorption_bands.keep_hitran_s(approximate_percentile=90)

# Automatically set up the methods to compute absorption coefficients
ws.propagation_matrix_agendaAuto()

# Set up a simple atmosphere
ws.surface_fieldPlanet(option="Earth")
ws.surface_field[pa.arts.SurfaceKey("t")] = 295.0
ws.atmospheric_fieldRead(
    toa=100e3, basename="planets/Earth/afgl/tropical/", missing_is_zero=1
)

# Set up geometry of observation
pos = [100e3, 0, 0]
los = [180.0, 0.0]
ws.ray_pathGeometric(pos=pos, los=los, max_step=1000.0)
ws.spectral_radianceClearskyEmission()

# --- Planck function ---
def planck_nu(nu, T):
    """Spectral radiance B_nu [W m^-2 sr^-1 Hz^-1]"""
    return (2*const.h*nu**3/const.c**2) / (np.exp(const.h*nu/(const.k*T)) - 1)

# Convert wavenumber grid to frequency (Hz)
freq_grid = ws.frequency_grid

# Temperatures to plot
temps = [200, 230, 260, 290]

# %% Show results
fig, ax = plt.subplots(figsize=(8,5))
ax.plot(kayser_grid, ws.spectral_radiance[:, 0], label="ARTS OLR spectrum", lw=1)

# Add Planck curves
for T in temps:
    B = planck_nu(freq_grid, T)
    ax.plot(kayser_grid, B, label=f"Planck {T} K", linestyle="--")

ax.set_xlabel("Wavenumber (cm$^{-1}$)")
ax.set_ylabel("Spectral radiance (W m$^{-2}$ sr$^{-1}$ Hz$^{-1}$)")
ax.set_title("Clear sky outgoing radiance with Planck curves")
ax.legend()
ax.grid(True)

if "ARTS_HEADLESS" not in os.environ:
    plt.show()

