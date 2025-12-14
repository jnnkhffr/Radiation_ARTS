# Ex7_optimized.py — Exercise 7 (fast version)


import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pyarts3 as pa
from Ex4_climate_model import climate_column

pa.data.download()

# Hilfsfunktionen
def build_atmosphere(Ts, Tcp, RH, N=100):
    """Baut ein ARTS-kompatibles Atmosphärenprofil."""
    p, T, x = climate_column(Ts=Ts, Tcp=Tcp, RH=RH, N=N)

    g = 9.80665
    Rd = 287.05
    Rv = 461.5
    epsilon = Rd / Rv

    e = x * p
    r = epsilon * e / (p - e)
    q = r / (1.0 + r)
    Tv = T * (1.0 + (Rv / Rd - 1.0) * q)

    z = np.zeros_like(p)
    for i in range(len(p)-1):
        Tv_bar = 0.5 * (Tv[i] + Tv[i+1])
        z[i+1] = z[i] + (Rd/g) * Tv_bar * np.log(p[i]/p[i+1])

    atm = xr.Dataset(
        {
            "t": (("lat","lon","alt"), T.reshape(1,1,-1)),
            "p": (("lat","lon","alt"), p.reshape(1,1,-1)),
            "H2O": (("lat","lon","alt"), x.reshape(1,1,-1)),
            "CO2": (("lat","lon","alt"), np.ones((1,1,len(p))) * 4e-4),
            "O3":  (("lat","lon","alt"), np.ones((1,1,len(p))) * 1e-6),
        },
        coords={"lat":[0.0], "lon":[0.0], "alt":z}
    )
    return atm, p, T, x


# Workspace nur EINMAL erzeugen
kayser_grid = np.linspace(1, 2000, 1000)

def init_workspace():
    ws = pa.workspace.Workspace()

    ws.frequency_grid = pa.arts.convert.kaycm2freq(kayser_grid)

    ws.absorption_speciesSet(species=[
        "H2O-161",
        "H2O-ForeignContCKDMT400",
        "H2O-SelfContCKDMT400",
        "CO2-626",
        "O3",
    ])

    ws.ReadCatalogData()
    ws.propagation_matrix_agendaAuto()

    # Oberfläche setzen (Atmosphäre kommt später)
    ws.surface_fieldPlanet(option="Earth")

    return ws


ws_global = init_workspace()

# OLR-Funktion (Workspace wird wiederverwendet)
def compute_OLR_spectrum_ARTS(ws, atm_dataset):
    ws.atmospheric_field = pa.data.to_atmospheric_field(atm_dataset)

    Ts = float(atm_dataset["t"].values[0,0,0])
    ws.surface_field[pa.arts.SurfaceKey("t")] = Ts

    # Ray path JETZT berechnen
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

# Exercise 7
def run_exercise7(Ts_baseline, ws):
    print(f"\n Running Exercise 7 for baseline Ts = {Ts_baseline} K")

    # Baseline
    atm_base, p, T_base, x_base = build_atmosphere(Ts=Ts_baseline, Tcp=200, RH=0.8)
    OLR_base = compute_OLR_spectrum_ARTS(ws, atm_base)

    # Case 1: Ts +1K, nur Oberfläche
    T1 = T_base.copy()
    T1[0] += 1.0
    atm1 = atm_base.copy()
    atm1["t"][:] = T1.reshape(1,1,-1)
    OLR1 = compute_OLR_spectrum_ARTS(ws, atm1)

    # Case 2: gesamtes T-Profil +1K
    T2 = T_base + 1.0
    atm2 = atm_base.copy()
    atm2["t"][:] = T2.reshape(1,1,-1)
    OLR2 = compute_OLR_spectrum_ARTS(ws, atm2)

    # Case 3: Ts+1K, T+H2O neu (RH konstant)
    atm3, _, _, _ = build_atmosphere(Ts=Ts_baseline+1, Tcp=200, RH=0.8)
    OLR3 = compute_OLR_spectrum_ARTS(ws, atm3)

    # Plot
    plt.figure(figsize=(10,6))
    plt.plot(kayser_grid, OLR1 - OLR_base, label="Case 1: only Ts +1K", lw=0.5)
    plt.plot(kayser_grid, OLR2 - OLR_base, label="Case 2: T-profile +1K", lw=0.5)
    plt.plot(kayser_grid, OLR3 - OLR_base, label="Case 3: T+H2O adjusted (RH const)", lw=0.5)
    plt.axhline(0, color="black", lw=0.5)
    plt.xlabel("Wavenumber [cm⁻¹]")
    plt.ylabel("ΔOLR [W/m²/cm⁻¹]")
    plt.title(f"Spectral ΔOLR for Ts baseline = {Ts_baseline} K")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Integrale
    d1 = integrate_OLR(kayser_grid, OLR1 - OLR_base)
    d2 = integrate_OLR(kayser_grid, OLR2 - OLR_base)
    d3 = integrate_OLR(kayser_grid, OLR3 - OLR_base)

    print(f"Integrated ΔOLR (Case 1): {d1} W/m²")
    print(f"Integrated ΔOLR (Case 2): {d2} W/m²")
    print(f"Integrated ΔOLR (Case 3): {d3} W/m²")

    return d1, d2, d3


# Run Exercise 7
results_290 = run_exercise7(290, ws_global)
results_300 = run_exercise7(300, ws_global)

print("\n=== Summary ===")
print("Baseline Ts = 290 K:", results_290)
print("Baseline Ts = 300 K:", results_300)

