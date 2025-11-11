"""
Clear-sky outgoing longwave radiation using ARTS,
with custom single-column atmosphere from climate_column().
"""

import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
#import faulthandler
#faulthandler.enable()
import matplotlib.pyplot as plt
import pyarts3 as pa

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

# --- Katalogdaten sicherstellen ---
pa.data.download()

# --- Konstanten für dein Säulenmodell ---
Ts = 290.0   # Oberflächentemperatur [K]
Tcp = 200.0  # Tropopausentemperatur [K]
RH = 0.8     # relative Feuchte

# --- Säulenprofil erzeugen ---
p, T, x = climate_column(Ts=Ts, Tcp=Tcp, RH=RH, N=100)
p_grid  = p.astype(float)      # Pa, fallend
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
# -> shapes are correct, error has to be with the content of the fields

print("p_grid decreasing:", np.all(np.diff(p_grid) < 0))
print("z_field increasing:", np.all(np.diff(z_field) > 0))
print("z_field min/max:", z_field.min(), z_field.max())

# --- XML-Dateien schreiben ---
write_vector_xml("p_grid.xml", p_grid)
write_vector_xml("t_field.xml", T_field)
write_arrayofvector_xml("vmr_field.xml", [H2O_vmr])
write_vector_xml("z_field.xml", z_field)

# --- Workspace ---
ws = pa.workspace.Workspace()

# Frequenzgitter (AscendingGrid: direkt NumPy-Array zuweisen)
kayser_grid = np.linspace(200, 1500, 100)  # cm^-1, reduzierter Bereich
ws.frequency_grid = pa.arts.convert.kaycm2freq(kayser_grid)

# Absorptionsspezies (erstmal nur H2O, stabiler)
#ws.absorption_speciesSet(species=["H2O-161, H2O-ForeignContCKDMT400, H2O-SelfContCKDMT400"])
ws.absorption_speciesSet(species=["H2O-161"])

# Katalogdaten laden
ws.ReadCatalogData()

# Absorptionsagenda
ws.propagation_matrix_agendaAuto()

# Oberfläche
ws.surface_fieldPlanet(option="Earth")
ws.surface_field[pa.arts.SurfaceKey("t")] = Ts

# Atmosphärenfelder per XML einlesen
print("before read in xml files")
print(os.path.abspath("z_field.xml"), os.path.exists("z_field.xml"))

import xml.etree.ElementTree as ET
try:
    ET.parse("z_field.xml")
    print("XML well-formed")
except Exception as e:
    print("XML parse error:", e)


ws.ReadXML("p_grid.xml") #works
print("p_grid finished")
ws.ReadXML("t_field.xml") #works
ws.ReadXML("vmr_field.xml") #works
ws.ReadXML("z_field.xml") #works
print("after read in xml files")
print("ws.atmosphere_numberOfVmiFields:", ws.atmosphere_numberOfVmiFields)

# Geometrie: Start knapp über Boden
pos = [z_field[0] + 1.0, 0.0, 0.0]
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







