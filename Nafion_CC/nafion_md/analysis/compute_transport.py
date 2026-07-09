#!/usr/bin/env python3
"""
compute_transport.py
---------------------
Post-processes LAMMPS output from in.nafion to get:

  1. Self-diffusion coefficients (Einstein relation) for H3O+ and water
  2. An approximate VEHICULAR proton conductivity via Nernst-Einstein
  3. The volumetric swelling ratio vs. a dry-membrane reference run

Usage:
    python3 compute_transport.py \
        --msd lammps/msd_vs_time.dat \
        --swelling-hydrated lammps/swelling.dat \
        --swelling-dry lammps_dry/swelling.dat \
        --n-carrier 40 --volume-A3 90886.0 --temp 300.0

IMPORTANT CAVEAT: classical (non-reactive) MD, as used in this pipeline,
only captures VEHICULAR diffusion of the intact H3O+ ion. Real Nafion
proton conductivity is dominated by Grotthuss (structural) proton
hopping, which is NOT captured here. The Nernst-Einstein estimate below
should therefore be read as a lower bound / vehicular-only contribution,
not a direct prediction of experimental conductivity. For a
Grotthuss-capable estimate, re-run the transport calculation with a
multistate empirical valence bond (MS-EVB) or ReaxFF potential instead
of this classical force field.
"""
import argparse
import numpy as np

KB = 1.380649e-23       # J/K
E_CHARGE = 1.602176634e-19  # C
NA = 6.02214076e23

def fit_msd_slope(t_ps, msd_A2, skip_frac=0.2):
    """Linear fit to the diffusive (late-time) regime of MSD(t)."""
    n = len(t_ps)
    i0 = int(n * skip_frac)
    A = np.vstack([t_ps[i0:], np.ones_like(t_ps[i0:])]).T
    slope, intercept = np.linalg.lstsq(A, msd_A2[i0:], rcond=None)[0]
    return slope  # units: A^2 / ps

def diffusion_from_slope(slope_A2_per_ps):
    """D = slope/6 for 3D MSD; convert A^2/ps -> cm^2/s."""
    D_A2_per_ps = slope_A2_per_ps / 6.0
    # 1 A^2/ps = 1e-16 cm^2 / 1e-12 s = 1e-4 cm^2/s
    return D_A2_per_ps * 1.0e-4  # cm^2/s

def load_msd_file(path):
    """LAMMPS ave/time output: '# step  msd_hydronium(A^2)  msd_water(A^2)'
    with header lines starting with '#'. Column 1 = timestep."""
    steps, msd_h, msd_w = [], [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            steps.append(float(parts[0]))
            msd_h.append(float(parts[1]))
            msd_w.append(float(parts[2]))
    return np.array(steps), np.array(msd_h), np.array(msd_w)

def nernst_einstein_conductivity(D_cm2_s, n_carrier, volume_A3, T_K, z=1):
    """sigma = sum_i (n_i z_i^2 e^2 / V k_B T) * D_i   [S/cm]"""
    V_cm3 = volume_A3 * 1.0e-24
    n_density = n_carrier / V_cm3  # ions/cm^3
    sigma = (n_density * (z * E_CHARGE) ** 2 * D_cm2_s) / (KB * T_K)  # S/cm (Gaussian-ish check below)
    # sigma units check: [1/cm^3][C^2][cm^2/s] / [J/K][K] = C^2/(cm*s*J) = (A*s)^2/(cm*s*J)
    # = A^2*s/(cm*J) = A*(A*s/J)= A*(C/J); since J = C*V -> C/J = 1/V -> units = A/V/cm = S/cm. OK.
    return sigma

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--msd', required=True, help='msd_vs_time.dat from in.nafion')
    ap.add_argument('--timestep-fs', type=float, default=1.0)
    ap.add_argument('--n-carrier', type=int, required=True,
                     help='number of hydronium ions (mobile proton carriers)')
    ap.add_argument('--volume-A3', type=float, required=True,
                     help='equilibrated box volume (A^3), from swelling.dat')
    ap.add_argument('--temp', type=float, default=300.0)
    ap.add_argument('--swelling-hydrated', default=None)
    ap.add_argument('--swelling-dry', default=None)
    args = ap.parse_args()

    steps, msd_h, msd_w = load_msd_file(args.msd)
    t_ps = steps * args.timestep_fs / 1000.0

    slope_h = fit_msd_slope(t_ps, msd_h)
    slope_w = fit_msd_slope(t_ps, msd_w)
    D_h = diffusion_from_slope(slope_h)
    D_w = diffusion_from_slope(slope_w)

    print(f"Hydronium (H3O+) diffusion coefficient: D = {D_h:.3e} cm^2/s")
    print(f"Water diffusion coefficient:             D = {D_w:.3e} cm^2/s")

    sigma = nernst_einstein_conductivity(D_h, args.n_carrier, args.volume_A3, args.temp)
    print(f"\nApproximate VEHICULAR proton conductivity (Nernst-Einstein):")
    print(f"  sigma = {sigma:.4e} S/cm   (carriers = hydronium only)")
    print(f"  NOTE: real Nafion sigma is typically 0.01-0.2 S/cm at full hydration,")
    print(f"        dominated by Grotthuss hopping NOT captured by this classical FF.")
    print(f"        Treat this number as a lower-bound / vehicular contribution only.")

    if args.swelling_hydrated and args.swelling_dry:
        def read_V(path):
            with open(path) as f:
                for line in f:
                    if 'V=' in line:
                        return float(line.strip().split('V=')[-1])
            raise ValueError(f"no V= found in {path}")
        V_wet = read_V(args.swelling_hydrated)
        V_dry = read_V(args.swelling_dry)
        swelling_pct = 100.0 * (V_wet - V_dry) / V_dry
        print(f"\nVolumetric swelling: V_dry={V_dry:.1f} A^3, V_wet={V_wet:.1f} A^3")
        print(f"  Swelling ratio = {swelling_pct:.2f} %")

if __name__ == '__main__':
    main()
