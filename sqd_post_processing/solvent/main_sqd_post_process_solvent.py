"""
================================================================================
main_sqd_post_process_solvent.py — SQD Post-Processing driver (solvent medium)
================================================================================

This is the entry-point for running the classical post-processing stage of
the Sample-Based Quantum Diagonalization (SQD) workflow on quantum-hardware
samples for the IEF-PCM solvent calculation.

Pipeline performed by this script:

    1.  Configure run parameters (geometry shift, charge, spin, active space,
        backend, …) and SQD hyper-parameters (SCRF iterations, S-CORE
        iterations per cycle, batches, samples, tolerances).
    2.  Load the converged classical SCF (PCM-decorated DFROHF) and
        active-space Hamiltonian integrals (h1e, h2e, ECORE) from the
        upstream RHF → AVAS → CCSD pipeline checkpoint.  The CASCI object
        ``mc`` carries the IEF-PCM reaction-field model.
    3.  Load the classical CCSD reference total energy at the requested shift.
    4.  Load the saved Qiskit Runtime sampler PUB result (bitstrings from the
        LUCJ ansatz on the chosen IBM Quantum backend).
    5.  Run the outer SCRF loop wrapping S-CORE + batched diagonalization:
        for each iteration the reaction-field potential V_solvent is
        recomputed from the SQD 1-RDM and embedded in h_core for the next
        cycle.
    6.  Pickle the per-SCRF result history (incl. inner S-CORE traces),
        write a lightweight JSON summary, print a tabular summary, and save
        both the SCRF and inner-S-CORE convergence plots.
================================================================================
"""

import os
import json
import pickle
import logging
from pathlib import Path

from utils_sqd_post_process_solvent import (
    load_converged_scf_and_integrals,
    load_reference_energy,
    load_hardware_results,
    run_sqd_solvent,
    plot_scrf_convergence,
    plot_inner_convergence,
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
medium = 'solvent'

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
# Option A (currently used): load SCF + integrals directly from the chk file.
# In solvent mode this returns a PCM-decorated CASCI object ``mc`` whose
# ``with_solvent`` attribute drives the SCRF loop later on.
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
# from utils_sqd_post_process_solvent import load_converged_scf, load_integrals
# mf, mc, mo_avas = load_converged_scf(
#     CLASSICAL_DATA_DIR, shift, medium,
#     as_label=as_label, num_orbitals=num_orbitals,
#     num_elec_a=num_elec_a, num_elec_b=num_elec_b,
# )
# h1e_cas, h2e_cas, ecore = load_integrals(
#     CLASSICAL_DATA_DIR, shift,
#     spin_number=spin_number, as_label=as_label, num_orbitals=num_orbitals,
# )

nuclear_repulsion_energy = ecore

# Classical reference (CCSD total energy) at this geometry shift
reference_energy = load_reference_energy(
    CLASSICAL_DATA_DIR, as_label, shift_array.index(shift)
)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  SQD hyper-parameters (SCRF outer loop + S-CORE inner loop)
# ══════════════════════════════════════════════════════════════════════════════
n_scrf_iter: int          = 5      # outer SCRF (reaction-field) iterations
max_iterations: int       = 5      # max inner S-CORE iters per SCRF cycle
energy_tol: float         = 1e-5
occupancies_tol: float    = 1e-3
num_batches: int          = 3
samples_per_batch: int    = 300
symmetrize_spin: bool     = not bool(spin_number)
carryover_threshold: float = 1e-4
max_cycle: int            = 200
spin_sq: float            = mf.spin_square()[0]
seed: int                 = 12345


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Main: validate, load HW samples, run SQD-SCRF, save & plot
# ══════════════════════════════════════════════════════════════════════════════
# ── Validate inputs ───────────────────────────────────────────────────────
if mc is None:
    raise ValueError(
        "PySCF CASCI object `mc` must be loaded.\n"
        "Set it in Section 2. It must have a solvent model attached\n"
        "(mc.with_solvent) and valid mo_coeff from AVAS."
    )

if not hasattr(mc, "with_solvent"):
    raise AttributeError(
        "The CASCI object does not have a solvent model.\n"
        "Create it with e.g.:  mc = solvent.ddCOSMO(mc)  or  mc.PCM()"
    )

# ── Ensure output directory exists ────────────────────────────────────────
Path(SQD_RESULTS_DIR).mkdir(parents=True, exist_ok=True)

# ── Load quantum samples ──────────────────────────────────────────────────
pub_result = load_hardware_results(
    RESULTS_DIR, num_qubits, shift, backend_name,
)
bit_array = pub_result.data.meas

# ── Run the SCRF + SQD post-processing ────────────────────────────────────
scrf_data, final_result = run_sqd_solvent(
    mc=mc,
    bit_array=bit_array,
    num_orbitals=num_orbitals,
    num_elec_a=num_elec_a,
    num_elec_b=num_elec_b,
    n_scrf_iter=n_scrf_iter,
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

# ── Save results ──────────────────────────────────────────────────────────
# Save the SCRF data (minus large arrays for portability)
out_pkl = os.path.join(
    SQD_RESULTS_DIR,
    f"sqd_solvent_scrf_{num_qubits}q_{backend_name}.pkl",
)
# Create a serializable version (strip rho_ao to save space if desired)
save_data = []
for d in scrf_data:
    save_record = {k: v for k, v in d.items() if k != "inner_history"}
    # Convert ndarrays to lists for JSON compatibility (optional)
    save_data.append(save_record)
with open(out_pkl, "wb") as f:
    pickle.dump(scrf_data, f)
log.info("Full SCRF result history saved to %s", out_pkl)

# Also save a lightweight JSON summary
out_json = os.path.join(
    SQD_RESULTS_DIR,
    f"sqd_solvent_summary_{num_qubits}q_{backend_name}.json",
)
json_summary = []
for d in scrf_data:
    json_summary.append({
        "scrf_iter":      d["scrf_iter"],
        "e_active":       d["e_active"],
        "ecore":          d["ecore"],
        "e_total":        d["e_total"],
        "e_solvent":      d["e_solvent"],
        "e_double_count": d["e_double_count"],
        "wall_time_s":    d["wall_time"],
    })
with open(out_json, "w") as f:
    json.dump(json_summary, f, indent=2)
log.info("JSON summary saved to %s", out_json)

# ── Print summary ─────────────────────────────────────────────────────────
print_summary(
    scrf_data,
    reference_energy,
    num_elec_a=num_elec_a,
    num_elec_b=num_elec_b,
    num_orbitals=num_orbitals,
    max_iterations=max_iterations,
    num_batches=num_batches,
    samples_per_batch=samples_per_batch,
)

# ── Plot convergence ──────────────────────────────────────────────────────
scrf_plot_path = os.path.join(
    SQD_RESULTS_DIR,
    f"sqd_solvent_convergence_{num_qubits}q_{backend_name}.png",
)
plot_scrf_convergence(scrf_data, reference_energy, save_path=scrf_plot_path)

inner_plot_path = os.path.join(
    SQD_RESULTS_DIR,
    f"sqd_solvent_inner_convergence_{num_qubits}q_{backend_name}.png",
)
plot_inner_convergence(scrf_data, save_path=inner_plot_path)

log.info("Done.")
