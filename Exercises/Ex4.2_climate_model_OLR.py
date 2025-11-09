"""
Clear-sky outgoing longwave radiation using ARTS,
with custom single-column atmosphere from climate_column() from another py-skript.
"""

import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyarts3 as pa

# Lese die Funktion aus 4.1 ein
from Ex4_climate_model import climate_column

# Hilfsfunktionen zum Schreiben von ARTS-XML
def write_vector_xml(filename, values):
    values = np.array(values, dtype=float)
    with open(filename, "w") as f:
        f.write('<?xml version="1.0"?>\n<ARTS>\n')
        f.write('  <Vector>\n')
        f.write(f'    <n>{len(values)}</n>\n')
        f.write('    <v>' + " ".join(map(str, values)) + '</v>\n')
        f.write('  </Vector>\n</ARTS>\n')

def write_arrayofvector_xml(filename, list_of_vectors):
    with open(filename, "w") as f:
        f.write('<?xml version="1.0"?>\n<ARTS>\n')
        f.write('  <ArrayOfVector>\n')
        for vec in list_of_vectors:
            values = np.array(vec, dtype=float)
            f.write('    <Vector>\n')
            f.write(f'      <n>{len(values)}</n>\n')
            f.write('      <v>' + " ".join(map(str, values)) + '</v>\n')
            f.write('    </Vector>\n')
        f.write('  </ArrayOfVector>\n</ARTS>\n')

# Konstanten
Ts = 290.0   # K
Tcp = 200.0  # K
RH = 0.8     # 80 %

# Atmosphärenprofile erzeugen
p, T, x = climate_column(Ts=Ts, Tcp=Tcp, RH=RH, N=100)

# ARTS erwartet aufsteigend
p_grid = np.flip(p).astype(float)
T_field = np.flip(T).astype(float)
H2O_vmr = np.flip(x).astype(float)

# XML-Dateien schreiben
write_vector_xml("p_grid.xml", p_grid)
write_vector_xml("t_field.xml", T_field)
write_arrayofvector_xml("vmr_field.xml", [H2O_vmr])

# Workspace
ws = pa.workspace.Workspace()

ws.ReadXML("p_grid.xml")
print("p_grid geladen")
ws.ReadXML("t_field.xml")
print("t_field geladen")
ws.ReadXML("vmr_field.xml")
print("vmr_field geladen")


# Frequenzgitter
kayser_grid = np.linspace(1, 2000, 200)  # cm^-1
ws.frequency_grid = pa.arts.convert.kaycm2freq(kayser_grid)

# Absorptionsspezies
ws.absorption_speciesSet(
    species=["H2O-161", "H2O-ForeignContCKDMT400", "H2O-SelfContCKDMT400"]
)

# Katalogdaten
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

# Geometrie
pos = [100e3, 0, 0]

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
