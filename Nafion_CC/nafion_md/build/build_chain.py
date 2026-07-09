#!/usr/bin/env python3
"""
build_chain.py
---------------
Builds one Nafion oligomer chain matching the structure in the reference
image:

    backbone:  -(CF2-CF2)x-(CF2-CF)y-
    side chain hung off each "y" backbone carbon:
        -O-[CF2-CF(CF3)]z-O-CF2-CF2-SO3H

Per the user's spec:
  * 10 monomer units total per chain (x + y = 10)
  * exactly ONE monomer carries a side chain (y = 1)  -> exactly one -SO3H
    per chain (the -SO3- group itself is NOT part of the bracketed repeat
    unit, matching the drawing)
  * z = 1 (standard Nafion side chain length: one -CF2-CF(CF3)- unit
    between the two ether oxygens; this is the classic Nafion side chain,
    as opposed to the longer "Aquivion"-type z>1 side chains)

The backbone is built as a finite, CF3-capped oligomer (20 backbone
carbons = 10 monomer units). This is the standard simplification used in
classical MD studies of short Nafion fragments -- an infinite/very long
real chain is approximated by a saturated, capped oligomer segment.

Output: chain.pdb  (single neutral, protonated -SO3H chain, all-atom,
        explicit H only on the acidic proton)
        chain_deprotonated.pdb (same chain but -SO3- with H removed,
        for the ionomer form used in the packed simulation box)
"""

from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

N_MONOMERS = 10        # total backbone monomer units in the chain
BRANCH_MONOMER_INDEX = 5  # which monomer (0-indexed) carries the side chain

def build_chain(deprotonate=False):
    mol = Chem.RWMol()
    conf_ids = []

    def add_atom(symbol, charge=0):
        a = Chem.Atom(symbol)
        if charge:
            a.SetFormalCharge(charge)
        return mol.AddAtom(a)

    # ---- backbone: 2 carbons per monomer, N_MONOMERS monomers ----
    n_backbone_C = 2 * N_MONOMERS
    backbone_idx = [add_atom('C') for _ in range(n_backbone_C)]
    for i in range(n_backbone_C - 1):
        mol.AddBond(backbone_idx[i], backbone_idx[i + 1], Chem.BondType.SINGLE)

    branch_carbon = backbone_idx[2 * BRANCH_MONOMER_INDEX + 1]  # 2nd C of that monomer

    # ---- fluorines on backbone ----
    for i, c in enumerate(backbone_idx):
        is_terminal = (i == 0 or i == n_backbone_C - 1)
        is_branch = (c == branch_carbon)
        n_F = 3 if is_terminal else (1 if is_branch else 2)
        for _ in range(n_F):
            f = add_atom('F')
            mol.AddBond(c, f, Chem.BondType.SINGLE)

    # ---- side chain: -O-CF2-CF(CF3)-O-CF2-CF2-SO3H ----
    O1 = add_atom('O')
    mol.AddBond(branch_carbon, O1, Chem.BondType.SINGLE)

    Ca = add_atom('C')                      # -CF2-
    mol.AddBond(O1, Ca, Chem.BondType.SINGLE)
    for _ in range(2):
        f = add_atom('F'); mol.AddBond(Ca, f, Chem.BondType.SINGLE)

    Cb = add_atom('C')                      # -CF(CF3)-
    mol.AddBond(Ca, Cb, Chem.BondType.SINGLE)
    f = add_atom('F'); mol.AddBond(Cb, f, Chem.BondType.SINGLE)

    Cc = add_atom('C')                      # -CF3 branch off Cb
    mol.AddBond(Cb, Cc, Chem.BondType.SINGLE)
    for _ in range(3):
        f = add_atom('F'); mol.AddBond(Cc, f, Chem.BondType.SINGLE)

    O2 = add_atom('O')
    mol.AddBond(Cb, O2, Chem.BondType.SINGLE)

    Cd = add_atom('C')                      # -CF2-
    mol.AddBond(O2, Cd, Chem.BondType.SINGLE)
    for _ in range(2):
        f = add_atom('F'); mol.AddBond(Cd, f, Chem.BondType.SINGLE)

    Ce = add_atom('C')                      # -CF2-
    mol.AddBond(Cd, Ce, Chem.BondType.SINGLE)
    for _ in range(2):
        f = add_atom('F'); mol.AddBond(Ce, f, Chem.BondType.SINGLE)

    S = add_atom('S')
    mol.AddBond(Ce, S, Chem.BondType.SINGLE)

    Od1 = add_atom('O')
    mol.AddBond(S, Od1, Chem.BondType.DOUBLE)
    Od2 = add_atom('O')
    mol.AddBond(S, Od2, Chem.BondType.DOUBLE)

    if deprotonate:
        Oh = add_atom('O', charge=-1)
        mol.AddBond(S, Oh, Chem.BondType.SINGLE)
    else:
        Oh = add_atom('O')
        mol.AddBond(S, Oh, Chem.BondType.SINGLE)
        H = add_atom('H')
        mol.AddBond(Oh, H, Chem.BondType.SINGLE)

    m = mol.GetMol()
    Chem.SanitizeMol(m)
    m = Chem.AddHs(m, addCoords=False)  # no other implicit H's exist, but keep API consistent

    ok = AllChem.EmbedMolecule(m, randomSeed=0xC0FFEE, useRandomCoords=True, maxAttempts=200)
    if ok != 0:
        raise RuntimeError("3D embedding failed")
    try:
        AllChem.UFFOptimizeMolecule(m, maxIters=2000)
    except Exception as e:
        print("UFF optimization warning:", e)

    return m

def write_pdb(mol, path, resname="NAF"):
    Chem.MolToPDBFile(mol, path, flavor=0)
    # patch residue name (RDKit defaults to UNL)
    with open(path) as f:
        txt = f.read()
    txt = txt.replace("UNL", resname[:3].rjust(3))
    with open(path, "w") as f:
        f.write(txt)

if __name__ == "__main__":
    m_prot = build_chain(deprotonate=False)
    print("Protonated chain atoms:", m_prot.GetNumAtoms(),
          "formula:", Chem.rdMolDescriptors.CalcMolFormula(m_prot))
    write_pdb(m_prot, "/home/moka/polymer_sims/Nafion_CC/nafion_md/build/chain_protonated.pdb")

    m_deprot = build_chain(deprotonate=True)
    print("Deprotonated chain atoms:", m_deprot.GetNumAtoms(),
          "formula:", Chem.rdMolDescriptors.CalcMolFormula(m_deprot))
    write_pdb(m_deprot, "/home/moka/polymer_sims/Nafion_CC/nafion_md/build/chain_deprotonated.pdb")

    # sanity check: exactly one S per chain
    n_S_prot = sum(1 for a in m_prot.GetAtoms() if a.GetSymbol() == 'S')
    n_S_deprot = sum(1 for a in m_deprot.GetAtoms() if a.GetSymbol() == 'S')
    assert n_S_prot == 1 and n_S_deprot == 1, "chain must have exactly one sulfonate group"
    print("Sanity check OK: exactly 1 sulfur (SO3H/SO3-) per chain, 10 backbone monomers.")
