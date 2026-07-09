#!/usr/bin/env python3
"""Generate single-molecule PDB templates used as packmol building blocks."""
import numpy as np

def write_pdb(path, resname, atoms):
    """atoms: list of (elem, name, x, y, z)"""
    with open(path, "w") as f:
        for i, (elem, name, x, y, z) in enumerate(atoms, start=1):
            f.write(f"HETATM{i:5d} {name:<4s} {resname:<3s}     1    "
                     f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {elem:>2s}\n")
        f.write("END\n")

# ---- SPC/E water: O-H = 1.0 A, H-O-H = 109.47 deg ----
r_OH = 1.0
theta = np.radians(109.47)
O = np.array([0.0, 0.0, 0.0])
H1 = np.array([r_OH * np.sin(theta/2),  r_OH * np.cos(theta/2), 0.0])
H2 = np.array([-r_OH * np.sin(theta/2), r_OH * np.cos(theta/2), 0.0])
write_pdb("/home/moka/polymer_sims/Nafion_CC/nafion_md/build/water.pdb", "SPC",
          [("O", "OW", *O), ("H", "H1", *H1), ("H", "H2", *H2)])

# ---- Hydronium H3O+: pyramidal, O-H = 0.98 A, H-O-H = 113 deg, C3v ----
r_OH = 0.98
ang = np.radians(113.0)
# place O at origin, apex along +z, 3 H's arranged symmetrically
h_polar = ang  # angle from +z axis to each O-H bond... approximate via umbrella height
# build via known pyramidal geometry: put 3 H in a ring below O
h_from_axis = np.radians(180.0 - 111.0)  # umbrella angle off the C3 axis (~ typical for H3O+)
z = -r_OH * np.cos(h_from_axis)
rho = r_OH * np.sin(h_from_axis)
Hs = []
for k in range(3):
    phi = np.radians(120.0 * k)
    Hs.append(np.array([rho*np.cos(phi), rho*np.sin(phi), z]))
write_pdb("/home/moka/polymer_sims/Nafion_CC/nafion_md/build/hydronium.pdb", "HYD",
          [("O", "OH", 0.0, 0.0, 0.0)] + [("H", f"H{k+1}", *Hs[k]) for k in range(3)])

print("wrote water.pdb and hydronium.pdb")
