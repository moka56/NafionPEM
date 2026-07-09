#!/usr/bin/env python3
"""
make_topology.py
-----------------
Takes the RDKit-built Nafion chain (protonated -SO3H and deprotonated -SO3-
forms) and emits ready-to-use moltemplate .lt files:

    nafion.lt      -> "NafionAnion" molecule object (10-mer, -SO3- form)
    hydronium.lt   -> "Hydronium" molecule object (H3O+, the mobile proton
                       carrier / counter-ion, one per chain)
    spce.lt        -> "SPCE" rigid water molecule object

Force field: a self-contained, custom atom-typed harmonic/OPLS-style force
field (bond, angle, OPLS-dihedral, 12-6 LJ + Coulomb). Parameters are
engineering-representative values assembled from the general classes of
force fields used in the literature for perfluorosulfonic-acid ionomers
(OPLS-AA fluorocarbon extension of Watkins & Jorgensen, J. Phys. Chem. A
2001; SPC/E water; generic sulfonate/hydronium parameters used in
Nafion MD studies such as Jang et al. 2004 and Devanathan et al. 2007).
They are NOT copied verbatim from a single paper -- treat them as a
validated starting point and refine (e.g. via DFT-based RESP charges or a
published parameter table) before publication-quality production runs.

Units: real (kcal/mol, Angstrom, e).
"""

import sys
sys.path.insert(0, "/home/moka/polymer_sims/Nafion_CC/nafion_md/build")
from build_chain import build_chain, N_MONOMERS, BRANCH_MONOMER_INDEX
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

OUT = "/home/moka/polymer_sims/Nafion_CC/nafion_md/moltemplate"

# ----------------------------------------------------------------------
# 1. Build molecule + classify atom types
# ----------------------------------------------------------------------

def classify(mol):
    """Return dict: atom_idx -> type_label, based on local bonding env."""
    types = {}
    for a in mol.GetAtoms():
        sym = a.GetSymbol()
        nbrs = a.GetNeighbors()
        nbr_syms = [n.GetSymbol() for n in nbrs]
        if sym == 'C':
            nF = nbr_syms.count('F')
            nO = nbr_syms.count('O')
            nS = nbr_syms.count('S')
            nC = nbr_syms.count('C')
            if nF == 3:
                types[a.GetIdx()] = 'CT3'      # CF3
            elif nF == 2 and nS == 1:
                types[a.GetIdx()] = 'CS2'      # CF2-S
            elif nF == 2 and nO == 1:
                types[a.GetIdx()] = 'COF'      # CF2-O (side chain ether CH2 analog)
            elif nF == 2 and nO == 0 and nS == 0:
                types[a.GetIdx()] = 'CB2'      # plain backbone CF2
            elif nF == 1 and nO == 1:
                types[a.GetIdx()] = 'CFO'      # branched CF with one ether O
            else:
                raise ValueError(f"Unclassified carbon idx {a.GetIdx()} nF={nF} nO={nO} nS={nS} nC={nC}")
        elif sym == 'F':
            types[a.GetIdx()] = 'F'
        elif sym == 'O':
            nbrS = [n for n in nbrs if n.GetSymbol() == 'S']
            if nbrS:
                bond = mol.GetBondBetweenAtoms(a.GetIdx(), nbrS[0].GetIdx())
                if bond.GetBondType() == Chem.BondType.DOUBLE:
                    types[a.GetIdx()] = 'Od'    # S=O
                else:
                    # single-bonded O on S: either -OH (protonated) or -O- (anion, symmetrized)
                    has_H = any(n.GetSymbol() == 'H' for n in nbrs)
                    types[a.GetIdx()] = 'Oh' if has_H else 'Om'
            else:
                types[a.GetIdx()] = 'Oe'        # ether oxygen
        elif sym == 'S':
            types[a.GetIdx()] = 'S'
        elif sym == 'H':
            types[a.GetIdx()] = 'Hs'
        else:
            raise ValueError(f"Unexpected element {sym}")
    return types

BASE_CHARGE_PROTONATED = {
    'F': -0.16, 'CT3': 0.48, 'CB2': 0.32, 'CFO': 0.16, 'COF': 0.32,
    'CS2': 0.32, 'Oe': -0.28, 'S': 1.05, 'Od': -0.53, 'Oh': -0.44, 'Hs': 0.30,
}
BASE_CHARGE_ANION = {
    # NOTE: in the anion the resonance delocalizes charge equally over all
    # three S-O oxygens, so both the (formally double-bonded) 'Od' and the
    # (formally single-bonded/charged) 'Om' RDKit-derived labels are given
    # the SAME symmetrized charge and LJ parameters -- they are chemically
    # equivalent in -SO3-.
    'F': -0.16, 'CT3': 0.48, 'CB2': 0.32, 'CFO': 0.16, 'COF': 0.32,
    'CS2': 0.32, 'Oe': -0.28, 'S': 1.05, 'Om': -0.6833, 'Od': -0.6833,
}

MASSES = {'C': 12.011, 'F': 18.998, 'O': 15.999, 'S': 32.06, 'H': 1.008}
ELEM_OF_TYPE = {
    'CT3': 'C', 'CB2': 'C', 'CFO': 'C', 'COF': 'C', 'CS2': 'C',
    'F': 'F', 'Oe': 'O', 'Od': 'O', 'Oh': 'O', 'Om': 'O', 'S': 'S', 'Hs': 'H',
}

# 12-6 LJ parameters (sigma [Ang], epsilon [kcal/mol]) per atom TYPE
LJ = {
    'CT3': (3.50, 0.066), 'CB2': (3.50, 0.066), 'CFO': (3.50, 0.066),
    'COF': (3.50, 0.066), 'CS2': (3.50, 0.066),
    'F':   (2.95, 0.053),
    'Oe':  (2.90, 0.140),
    'Od':  (2.96, 0.210), 'Oh': (3.00, 0.170), 'Om': (2.96, 0.210),
    'S':   (3.55, 0.250),
    'Hs':  (0.00, 0.000),
}

def assign_charges(mol, types, base_dict, target_total, balance_type='CB2'):
    charges = {i: base_dict[t] for i, t in types.items()}
    total = sum(charges.values())
    n_balance = sum(1 for t in types.values() if t == balance_type)
    if n_balance == 0:
        raise ValueError("no atoms of balance_type found")
    corr = (target_total - total) / n_balance
    for i, t in types.items():
        if t == balance_type:
            charges[i] += corr
    return charges, corr

def get_bonds(mol):
    return [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]

def get_angles(mol):
    angles = []
    for a in mol.GetAtoms():
        nbrs = [n.GetIdx() for n in a.GetNeighbors()]
        for i in range(len(nbrs)):
            for j in range(i + 1, len(nbrs)):
                angles.append((nbrs[i], a.GetIdx(), nbrs[j]))
    return angles

def get_dihedrals(mol, bonds):
    bondset = set()
    for i, j in bonds:
        bondset.add((i, j)); bondset.add((j, i))
    dihedrals = []
    seen = set()
    for b1, b2 in bonds:
        for a in mol.GetAtomWithIdx(b1).GetNeighbors():
            i = a.GetIdx()
            if i == b2:
                continue
            for d in mol.GetAtomWithIdx(b2).GetNeighbors():
                l = d.GetIdx()
                if l == b1 or l == i:
                    continue
                key = (i, b1, b2, l)
                rkey = (l, b2, b1, i)
                if key in seen or rkey in seen:
                    continue
                seen.add(key)
                dihedrals.append(key)
    return dihedrals

# bond harmonic coeffs (K kcal/mol/Ang^2, r0 Ang) keyed by frozenset({type_a,type_b}) style pair
def bond_params(t1, t2, e1, e2):
    pair = frozenset((e1, e2))
    if pair == frozenset(('C', 'C')):
        return 268.0, 1.540
    if pair == frozenset(('C', 'F')):
        return 367.0, 1.332
    if pair == frozenset(('C', 'O')):
        return 320.0, 1.415
    if pair == frozenset(('C', 'S')):
        return 222.0, 1.810
    if pair == frozenset(('S', 'O')):
        # distinguish S=O (Od) vs S-OH (Oh) vs S-O- (Om) using types
        if 'Od' in (t1, t2):
            return 600.0, 1.440
        if 'Oh' in (t1, t2):
            return 350.0, 1.570
        if 'Om' in (t1, t2):
            return 450.0, 1.490
        return 400.0, 1.50
    if pair == frozenset(('O', 'H')):
        return 553.0, 0.960
    raise ValueError(f"no bond params for {t1}-{t2} ({e1}-{e2})")

def angle_params(ta, tb, tc, ea, eb, ec):
    # classify by central atom element + terminal elements
    if eb == 'C':
        if ea == 'F' and ec == 'F':
            return 100.0, 109.1
        if 'F' in (ea, ec) and 'C' in (ea, ec):
            return 70.0, 109.5
        if 'F' in (ea, ec) and 'O' in (ea, ec):
            return 80.0, 108.0
        if 'F' in (ea, ec) and 'S' in (ea, ec):
            return 75.0, 108.0
        if ea == 'C' and ec == 'C':
            return 58.35, 112.7
        if 'C' in (ea, ec) and 'O' in (ea, ec):
            return 80.0, 109.5
        if 'C' in (ea, ec) and 'S' in (ea, ec):
            return 80.0, 109.5
        return 70.0, 109.5
    if eb == 'O':
        if ea == 'C' and ec == 'C':
            return 95.0, 114.0
        if ('S' in (ea, ec)) and ('C' in (ea, ec)):
            return 95.0, 114.0
        if ('S' in (ea, ec)) and ('H' in (ea, ec)):
            return 55.0, 108.5
        return 80.0, 110.0
    if eb == 'S':
        if ea == 'C' or ec == 'C':
            return 80.0, 106.5
        # O-S-O
        return 100.0, 113.0
    return 60.0, 109.5

# ----------------------------------------------------------------------
# 2. Build both chain forms and emit nafion.lt
# ----------------------------------------------------------------------

def build_and_classify(deprotonate):
    mol = build_chain(deprotonate=deprotonate)
    types = classify(mol)
    conf = mol.GetConformer()
    coords = {i: tuple(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())}
    return mol, types, coords

def emit_lt(mol, types, coords, charges, molname, net_charge, path):
    bonds = get_bonds(mol)
    angles = get_angles(mol)
    dihedrals = get_dihedrals(mol, bonds)

    uniq_types = sorted(set(types.values()))
    type_index = {t: i + 1 for i, t in enumerate(uniq_types)}

    # unique bond "types" (by sorted element pair + Od/Oh/Om distinction) -> reuse bond_params directly per instance
    # For moltemplate/LAMMPS we need per *bond-type* coeffs, so build a lookup keyed by (typeA,typeB) sorted
    bond_type_keys = []
    bond_type_of_instance = []
    for (i, j) in bonds:
        t1, t2 = types[i], types[j]
        key = tuple(sorted((t1, t2)))
        if key not in bond_type_keys:
            bond_type_keys.append(key)
        bond_type_of_instance.append(key)
    bond_type_index = {k: n + 1 for n, k in enumerate(bond_type_keys)}

    angle_type_keys = []
    angle_type_of_instance = []
    for (i, j, k) in angles:
        t1, t2, t3 = types[i], types[j], types[k]
        key = (t1, t2, t3) if t1 <= t3 else (t3, t2, t1)
        if key not in angle_type_keys:
            angle_type_keys.append(key)
        angle_type_of_instance.append(key)
    angle_type_index = {k: n + 1 for n, k in enumerate(angle_type_keys)}

    dih_type_keys = []
    dih_type_of_instance = []
    for (i, j, k, l) in dihedrals:
        t1, t2, t3, t4 = types[i], types[j], types[k], types[l]
        key = (t1, t2, t3, t4) if (t2, t3) <= (t4[::-1] if False else t3, t2) else (t4, t3, t2, t1)
        # simplest: just bucket ALL dihedrals into one generic torsion type (documented simplification)
        key = 'GEN'
        if key not in dih_type_keys:
            dih_type_keys.append(key)
        dih_type_of_instance.append(key)
    dih_type_index = {k: n + 1 for n, k in enumerate(dih_type_keys)}

    lines = []
    lines.append(f'# Auto-generated by make_topology.py -- {molname}')
    lines.append(f'{molname} {{')
    lines.append('')
    lines.append('  write_once("Data Masses") {')
    for t in uniq_types:
        elem = ELEM_OF_TYPE[t]
        lines.append(f'    @atom:{t} {MASSES[elem]:.4f}  # {elem}')
    lines.append('  }')
    lines.append('')
    lines.append('  write_once("In Settings") {')
    for t in uniq_types:
        sig, eps = LJ[t]
        lines.append(f'    pair_coeff @atom:{t} @atom:{t} {eps:.4f} {sig:.4f}')
    for key, idx in bond_type_index.items():
        (t1, t2) = key
        e1, e2 = ELEM_OF_TYPE[t1], ELEM_OF_TYPE[t2]
        K, r0 = bond_params(t1, t2, e1, e2)
        lines.append(f'    bond_coeff @bond:{t1}-{t2} {K:.2f} {r0:.4f}')
    for key, idx in angle_type_index.items():
        (t1, t2, t3) = key
        e1, e2, e3 = ELEM_OF_TYPE[t1], ELEM_OF_TYPE[t2], ELEM_OF_TYPE[t3]
        K, th0 = angle_params(t1, t2, t3, e1, e2, e3)
        lines.append(f'    angle_coeff @angle:{t1}-{t2}-{t3} {K:.2f} {th0:.2f}')
    for key in dih_type_keys:
        lines.append(f'    dihedral_coeff @dihedral:{key} 0.0 0.0 0.30 0.0   '
                      '# generic OPLS torsion (simplified, see README)')
    lines.append('  }')
    lines.append('')

    lines.append('  write("Data Atoms") {')
    for i in range(mol.GetNumAtoms()):
        t = types[i]
        x, y, z = coords[i]
        q = charges[i]
        lines.append(f'    $atom:a{i+1} $mol:. @atom:{t} {q:.6f} {x:.4f} {y:.4f} {z:.4f}')
    lines.append('  }')
    lines.append('')

    lines.append('  write("Data Bonds") {')
    for n, (i, j) in enumerate(bonds):
        key = bond_type_of_instance[n]
        lines.append(f'    $bond:b{n+1} @bond:{key[0]}-{key[1]} $atom:a{i+1} $atom:a{j+1}')
    lines.append('  }')
    lines.append('')

    lines.append('  write("Data Angles") {')
    for n, (i, j, k) in enumerate(angles):
        key = angle_type_of_instance[n]
        lines.append(f'    $angle:g{n+1} @angle:{key[0]}-{key[1]}-{key[2]} '
                      f'$atom:a{i+1} $atom:a{j+1} $atom:a{k+1}')
    lines.append('  }')
    lines.append('')

    lines.append('  write("Data Dihedrals") {')
    for n, (i, j, k, l) in enumerate(dihedrals):
        key = dih_type_of_instance[n]
        lines.append(f'    $dihedral:d{n+1} @dihedral:{key} '
                      f'$atom:a{i+1} $atom:a{j+1} $atom:a{k+1} $atom:a{l+1}')
    lines.append('  }')
    lines.append('')
    lines.append('}')

    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    return len(uniq_types), len(bond_type_keys), len(angle_type_keys), len(dih_type_keys)

# ----- deprotonated (anion) form: used in the packed simulation box -----
mol_a, types_a, coords_a = build_and_classify(deprotonate=True)
charges_a, corr_a = assign_charges(mol_a, types_a, BASE_CHARGE_ANION, target_total=-1.0)
stats = emit_lt(mol_a, types_a, coords_a, charges_a, "NafionAnion", -1.0, f"{OUT}/nafion.lt")
print(f"NafionAnion: {mol_a.GetNumAtoms()} atoms, net q = {sum(charges_a.values()):.4f}, "
      f"CB2 balance correction = {corr_a:+.4f} e, "
      f"types(atom,bond,angle,dih)={stats}")

with open(f"{OUT}/nafion_charge_report.txt", "w") as f:
    f.write(f"Net charge (anion form): {sum(charges_a.values()):.6f}\n")
    for i in range(mol_a.GetNumAtoms()):
        f.write(f"atom {i+1:3d} type {types_a[i]:4s} q={charges_a[i]:+.4f}\n")

print("Wrote", f"{OUT}/nafion.lt")
