#!/usr/bin/env bash
# ==============================================================================
# run_all.sh -- rebuild the whole Nafion PEM simulation from scratch
#
# Requires: python3 + rdkit (pip install rdkit), packmol, moltemplate
#           (pip install moltemplate), lammps (with KSPACE, MOLECULE,
#           RIGID/SHAKE support -- the stock Ubuntu "lammps" apt package
#           or a from-source build both work).
# ==============================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "== 1/5  Building the Nafion chain + water + hydronium PDB templates =="
python3 "$ROOT/build/build_chain.py"
python3 "$ROOT/build/build_water_hydronium.py"

echo "== 2/5  Generating force-field-typed moltemplate .lt files =="
python3 "$ROOT/build/make_topology.py"

echo "== 3/5  Packing the simulation box with packmol =="
cd "$ROOT/packmol"
packmol < pack_system.inp
# optional dry-membrane reference box (for swelling comparison):
packmol < pack_system_dry.inp

echo "== 4/5  Building LAMMPS topology/coefficients with moltemplate =="
cd "$ROOT/moltemplate"
cp "$ROOT/packmol/packed_system.pdb" .
moltemplate.sh -pdb packed_system.pdb system.lt
cleanup_moltemplate.sh 2>/dev/null || true

mkdir -p dry
cd dry
cp ../nafion.lt ../hydronium.lt "$ROOT/packmol/packed_system_dry.pdb" .
moltemplate.sh -pdb packed_system_dry.pdb system_dry.lt
cd ..

echo "== 5/5  Ready to run LAMMPS =="
cp system.data system.in.settings "$ROOT/lammps/"
echo "Hydrated system:  cd $ROOT/lammps  &&  lmp -in in.nafion"
echo "Dry reference:    cd $ROOT/moltemplate/dry  &&  lmp -in in.nafion_dry"
echo ""
echo "Then post-process with:"
echo "  python3 $ROOT/analysis/compute_transport.py --msd $ROOT/lammps/msd_vs_time.dat \\"
echo "      --n-carrier 40 --volume-A3 <V_eq from swelling.dat> --temp 300.0 \\"
echo "      --swelling-hydrated $ROOT/lammps/swelling.dat \\"
echo "      --swelling-dry $ROOT/moltemplate/dry/swelling.dat"
