#!/usr/bin/env bash
# ==============================================================================
# run_full_pipeline.sh -- ONE command that does everything:
#   1. builds the chain/water/hydronium/force-field
#   2. packs + runs the HYDRATED system (in.nafion)
#   3. packs + runs the DRY reference system (in.nafion_dry)
#   4. prints diffusion coefficients, Nernst-Einstein conductivity, and
#      swelling % all together at the end
#
# Both LAMMPS runs use the 12-hour-budget stage lengths (100ps NVT / 400ps
# NPT / 300ps production) with early checkpointing (restart.a/restart.b
# from right after minimization) so a crash doesn't lose all progress.
#
# Run this every time you want a fresh full-cycle result:
#   ./run_full_pipeline.sh
# Recommended: run inside tmux so it survives window closures/sleep --
#   tmux new -s nafion
#   ./run_full_pipeline.sh
#   (Ctrl+B, D to detach; tmux attach -t nafion to check back in)
# ==============================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

N_CHAIN=40
LAMBDA=22

echo "############################################################"
echo "# STAGE A: shared build (chain, water, hydronium, topology)"
echo "############################################################"
rm -f lammps/swelling.dat lammps/log.lammps lammps/*.lammpstrj
rm -f lammps/system.data lammps/system.in.settings
rm -f lammps/restart.a lammps/restart.b lammps/system_checkpoint1.*
rm -f moltemplate/system.data moltemplate/system.in.settings
rm -f moltemplate/dry/swelling.dat moltemplate/dry/log.lammps
rm -f moltemplate/dry/system_dry.data moltemplate/dry/system_dry.in.settings
rm -f moltemplate/dry/restart.a moltemplate/dry/restart.b moltemplate/dry/system_dry_checkpoint1.*

grep "N_MONOMERS =\|BRANCH_MONOMER_INDEX =" build/build_chain.py
python3 build/build_chain.py
python3 build/build_water_hydronium.py
python3 build/make_topology.py

echo ""
echo "############################################################"
echo "# STAGE B: compute box sizes (hydrated + dry) and write inputs"
echo "############################################################"
python3 - "$ROOT" "$N_CHAIN" "$LAMBDA" <<'PYEOF'
import sys
sys.path.insert(0, sys.argv[1] + "/build")
from build_chain import build_chain
from rdkit.Chem import Descriptors

ROOT = sys.argv[1]
N_CHAIN = int(sys.argv[2])
LAMBDA = int(sys.argv[3])

m = build_chain(deprotonate=True)
mw_chain = Descriptors.MolWt(m)
mw_h3o, mw_h2o = 19.023, 18.015

# --- hydrated box ---
N_HYD = N_CHAIN
N_WAT = N_CHAIN * LAMBDA
mass_wet_g = (N_CHAIN*mw_chain + N_HYD*mw_h3o + N_WAT*mw_h2o) / 6.02214076e23
L_wet = round((mass_wet_g / 0.9 * 1e24) ** (1/3), 2)   # loose pack, NPT compresses

# --- dry box (no water) ---
mass_dry_g = (N_CHAIN*mw_chain + N_HYD*mw_h3o) / 6.02214076e23
L_dry = round((mass_dry_g / 1.1 * 1e24) ** (1/3), 2)   # loose pack, denser dry target

print(f"HYDRATED: N_chain={N_CHAIN} N_hyd={N_HYD} N_wat={N_WAT} box L={L_wet} A")
print(f"DRY:      N_chain={N_CHAIN} N_hyd={N_HYD} N_wat=0    box L={L_dry} A")

with open(f"{ROOT}/packmol/pack_system.inp", "w") as f:
    f.write(f"""tolerance 2.0
nloop 300
filetype pdb
output packed_system.pdb
seed 12345

structure {ROOT}/build/chain_deprotonated.pdb
  number {N_CHAIN}
  inside cube 0. 0. 0. {L_wet}
end structure

structure {ROOT}/build/hydronium.pdb
  number {N_HYD}
  inside cube 0. 0. 0. {L_wet}
end structure

structure {ROOT}/build/water.pdb
  number {N_WAT}
  inside cube 0. 0. 0. {L_wet}
end structure
""")

with open(f"{ROOT}/packmol/pack_system_dry.inp", "w") as f:
    f.write(f"""tolerance 2.0
nloop 300
filetype pdb
output packed_system_dry.pdb
seed 12345

structure {ROOT}/build/chain_deprotonated.pdb
  number {N_CHAIN}
  inside cube 0. 0. 0. {L_dry}
end structure

structure {ROOT}/build/hydronium.pdb
  number {N_HYD}
  inside cube 0. 0. 0. {L_dry}
end structure
""")

with open(f"{ROOT}/moltemplate/system.lt", "w") as f:
    f.write(f"""import "nafion.lt"
import "hydronium.lt"
import "spce.lt"

chains    = new NafionAnion [{N_CHAIN}]
hydronium = new Hydronium   [{N_HYD}]
water     = new SPCE        [{N_WAT}]

write_once("Data Boundary") {{
  0.0  {L_wet}  xlo xhi
  0.0  {L_wet}  ylo yhi
  0.0  {L_wet}  zlo zhi
}}
""")

with open(f"{ROOT}/moltemplate/dry/system_dry.lt", "w") as f:
    f.write(f"""import "nafion.lt"
import "hydronium.lt"

chains    = new NafionAnion [{N_CHAIN}]
hydronium = new Hydronium   [{N_HYD}]

write_once("Data Boundary") {{
  0.0  {L_dry}  xlo xhi
  0.0  {L_dry}  ylo yhi
  0.0  {L_dry}  zlo zhi
}}
""")

print("Wrote pack_system.inp, pack_system_dry.inp, system.lt, system_dry.lt")
PYEOF

echo ""
echo "############################################################"
echo "# STAGE C: pack both boxes with packmol"
echo "############################################################"
cd packmol
packmol < pack_system.inp
packmol < pack_system_dry.inp
cd ..

echo ""
echo "############################################################"
echo "# STAGE D: build LAMMPS topology with moltemplate (both)"
echo "############################################################"
cd moltemplate
cp ../packmol/packed_system.pdb .
moltemplate.sh -pdb packed_system.pdb system.lt

mkdir -p dry
cd dry
cp ../nafion.lt ../hydronium.lt ../../packmol/packed_system_dry.pdb .
moltemplate.sh -pdb packed_system_dry.pdb system_dry.lt
cd ../..

echo ""
echo "############################################################"
echo "# STAGE E: verify SHAKE bond/angle type IDs (hydrated only"
echo "#          needs this -- dry has no water)"
echo "############################################################"
SHAKE_CHECK=$(grep -c "bond_coeff 18 1000.00\|angle_coeff 36 1000.00" moltemplate/system.in.settings || true)
if [ "$SHAKE_CHECK" -lt 2 ]; then
  echo "!!! WARNING: SHAKE bond/angle type IDs no longer match 18/36 !!!"
  grep "1000.00" moltemplate/system.in.settings
  echo "    Update the 'fix rigidwat ... shake ... b 18 a 36' line in"
  echo "    lammps/in.nafion to match before proceeding. Aborting."
  exit 1
fi
echo "    OK: bond 18 / angle 36 confirmed for hydrated system."

cp moltemplate/system.data moltemplate/system.in.settings lammps/
sed -i "s/group           nafion  molecule <= [0-9]*/group           nafion  molecule <= ${N_CHAIN}/" lammps/in.nafion

NCORES=$(nproc)

echo ""
echo "############################################################"
echo "# STAGE F: run HYDRATED system (minimize -> NVT -> NPT ->"
echo "#          production checkpoint)"
echo "############################################################"
cd lammps
mpirun --oversubscribe -np "$NCORES" lmp -in in.nafion
cd ..

echo ""
echo "############################################################"
echo "# STAGE G: run DRY reference system"
echo "############################################################"
cd moltemplate/dry
mpirun --oversubscribe -np "$NCORES" lmp -in in.nafion_dry
cd ../..

echo ""
echo "############################################################"
echo "# STAGE H: results -- diffusion, Nernst-Einstein conductivity,"
echo "#          swelling %"
echo "############################################################"
V_HYD=$(grep "V=" lammps/swelling.dat | tail -1 | sed 's/.*V=//')
V_DRY=$(grep "V=" moltemplate/dry/swelling.dat | tail -1 | sed 's/.*V=//')

python3 analysis/compute_transport.py \
  --msd lammps/msd_vs_time.dat \
  --n-carrier "$N_CHAIN" \
  --volume-A3 "$V_HYD" \
  --temp 300.0 \
  --swelling-hydrated lammps/swelling.dat \
  --swelling-dry moltemplate/dry/swelling.dat

echo ""
echo "Done. (Reminder: these are the 12h-budget SHORTENED-stage results --"
echo "preliminary, not publication-quality -- use continue_production.in"
echo "to extend further toward the collaborator's recommended lengths.)"
