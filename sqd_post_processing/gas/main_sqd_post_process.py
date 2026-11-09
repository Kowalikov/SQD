"""
================================================================================
main_sqd_post_process.py — SQD Post-Processing driver (gas medium)
================================================================================

This is the entry-point for running the classical post-processing stage of
the Sample-Based Quantum Diagonalization (SQD) workflow on quantum-hardware
samples for the gas-phase calculation.

Pipeline performed by this script:

    1.  Configure run parameters (geometry shift, charge, spin, active space,
        backend, ...) and SQD hyper-parameters (batches, samples, tolerances).
    2.  Load the converged classical SCF (ROHF + DF) and active-space
        Hamiltonian integrals (h1e, h2e, ECORE) from the upstream
        RHF → AVAS → CCSD pipeline checkpoint.
    3.  Load the classical CCSD reference total energy at the requested shift.
    4.  Load the saved Qiskit Runtime sampler PUB result (bitstrings from the
        LUCJ ansatz on the chosen IBM Quantum backend).
    5.  Run the S-CORE + batched subspace diagonalization loop.
    6.  Pickle the per-iteration result history, print a summary table, and
        save the convergence + orbital-occupancy diagnostic plot.
================================================================================
"""

import os
import pickle
import logging
from pathlib import Path

from utils_sqd_post_process import (
    load_converged_scf_and_integrals,
    load_reference_energy,
    load_hardware_results,
    run_sqd,
    plot_results,
    print_summary,
)

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Initial parameters
# ══════════════════════════════════════════════════════════════════════════════
shift_array = ['-0.15', '0.0', '0.15', '0.3', '0.45', '0.6',
               '0.75', '0.9', '1.05', '1.2']
shift = '0.0'

charge = 3
spin_number = 4
medium = 'gas'

num_orbitals = 17
num_elec_a, num_elec_b = 13, 9
num_qubits = num_orbitals * 2

backend_name = 'ibm_pittsburgh'

CLASSICAL_DATA_DIR = (
    f'../../classical_methods/{medium}/charge{charge}_spin{spin_number}'
)
RESULTS_DIR = (
    f'../../sampler_runs/{medium}/charge{charge}_spin{spin_number}/'
    f'{num_qubits}qubits'
)
as_label = 'norb_%d_nelec_%d_%d' % (num_orbitals, num_elec_a, num_elec_b)

SQD_RESULTS_DIR = (
    f'charge{charge}_spin{spin_number}/{num_qubits}qubits/shift_{shift}'
)
Path(SQD_RESULTS_DIR).mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Load classical (pre-quantum) inputs
# ══════════════════════════════════════════════════════════════════════════════
# Option A (currently used): load SCF + integrals directly from the chk file
mf, mc, h1e_cas, h2e_cas, ecore = load_converged_scf_and_integrals(
    CLASSICAL_DATA_DIR,
    shift,
    medium,
    as_label=as_label,
    num_orbitals=num_orbitals,
    num_elec_a=num_elec_a,
    num_elec_b=num_elec_b,
)

# Option B (alternative, fcidump-based) — uncomment if preferred:
# from utils_sqd_post_process import load_converged_scf, load_integrals
# mf = load_converged_scf(
#     CLASSICAL_DATA_DIR, shift, medium,
#     as_label=as_label, num_orbitals=num_orbitals,
#     num_elec_a=num_elec_a, num_elec_b=num_elec_b,
# )
# h1e_cas, h2e_cas, ecore = load_integrals(
#     CLASSICAL_DATA_DIR, shift,
#     spin_number=spin_number, as_label=as_label, num_orbitals=num_orbitals,
# )

nuclear_repulsion_energy = ecore  # mf.mol.energy_nuc()

# Classical reference (CCSD total energy) at this geometry shift
reference_energy = load_reference_energy(
    CLASSICAL_DATA_DIR, as_label, shift_array.index(shift)
)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  SQD hyper-parameters
# ══════════════════════════════════════════════════════════════════════════════
max_iterations: int       = 5
energy_tol: float         = 1e-5
occupancies_tol: float    = 1e-3
num_batches: int          = 5
samples_per_batch: int    = 200
symmetrize_spin: bool     = not bool(spin_number)
carryover_threshold: float = 1e-4
max_cycle: int            = 200
spin_sq: float            = mf.spin_square()[0]
seed: int                 = 12345


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Main: validate, load HW samples, run SQD, save & plot
# ══════════════════════════════════════════════════════════════════════════════
# ── Validate inputs ────────────────────────────────────────────────────────
if h1e_cas is None or h2e_cas is None:
    raise ValueError(
        "Active-space integrals h1e_cas and h2e_cas must be set.\n"
        "Load them from the classical checkpoint in Section 2 above."
    )

if nuclear_repulsion_energy == 0.0:
    log.warning(
        "nuclear_repulsion_energy is 0.0 — did you forget to set it? "
        "The reported total energies will be missing the nuclear + core offset."
    )

# ── Load quantum samples ───────────────────────────────────────────────────
pub_result = load_hardware_results(
    RESULTS_DIR, num_qubits, shift, backend_name
)
bit_array = pub_result.data.meas

# ── Run the S-CORE + diagonalization ───────────────────────────────────────
result, result_history, wall_time = run_sqd(
    h1e=h1e_cas,
    h2e=h2e_cas,
    bit_array=bit_array,
    num_orbitals=num_orbitals,
    num_elec_a=num_elec_a,
    num_elec_b=num_elec_b,
    nuclear_repulsion_energy=nuclear_repulsion_energy,
    max_iterations=max_iterations,
    energy_tol=energy_tol,
    occupancies_tol=occupancies_tol,
    num_batches=num_batches,
    samples_per_batch=samples_per_batch,
    symmetrize_spin=symmetrize_spin,
    carryover_threshold=carryover_threshold,
    max_cycle=max_cycle,
    spin_sq=spin_sq,
    seed=seed,
)

# ── Save full result history ───────────────────────────────────────────────
out_pkl = os.path.join(
    SQD_RESULTS_DIR,
    f"sqd_history_qubits_{num_qubits}_shift_{shift}_{backend_name}.pkl",
)
with open(out_pkl, "wb") as f:
    pickle.dump(result_history, f)
log.info("Result history saved to %s", out_pkl)

# ── Print summary + plot ───────────────────────────────────────────────────
print_summary(
    result,
    result_history,
    nuclear_repulsion_energy,
    reference_energy,
    wall_time,
    num_elec_a=num_elec_a,
    num_elec_b=num_elec_b,
    num_orbitals=num_orbitals,
    num_batches=num_batches,
    samples_per_batch=samples_per_batch,
)

plot_path = os.path.join(
    SQD_RESULTS_DIR,
    f"sqd_convergence_qubits_{num_qubits}_shift_{shift}_{backend_name}.png",
)
plot_results(
    result,
    result_history,
    nuclear_repulsion_energy,
    reference_energy,
    save_path=plot_path,
)
