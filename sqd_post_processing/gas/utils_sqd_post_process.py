"""
================================================================================
utils_sqd_post_process.py — Helper functions for SQD Post-Processing (gas medium)
================================================================================

PYSCF VERSION=2.9.0
This module collects all helper functions used by the SQD post-processing
pipeline:

    Loading classical inputs
    ------------------------
    - load_converged_scf
    - load_converged_scf_and_integrals
    - load_integrals
    - load_reference_energy

    Loading quantum hardware data
    -----------------------------
    - load_hardware_results

    SQD driver
    ----------
    - run_sqd

    Analysis & plotting
    -------------------
    - analyze_convergence
    - plot_results
    - print_summary
================================================================================
"""

import os
import json
import time
import logging
from functools import partial
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import pyscf
from pyscf import gto, scf, mp, mcscf, cc, ci, tools, ao2mo
from pyscf.scf import chkfile
from pyscf import lib
from pyscf.tools import fcidump

from qiskit_ibm_runtime import RuntimeDecoder

from qiskit_addon_sqd.fermion import (
    SCIResult,
    diagonalize_fermionic_hamiltonian,
    solve_sci_batch,
)

# ──────────────────────────────────────────────────────────────────────────────
# Module logger (configuration is set up by the caller, e.g. main script)
# ──────────────────────────────────────────────────────────────────────────────
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Loading classical (pre-quantum) inputs
# ══════════════════════════════════════════════════════════════════════════════
def load_converged_scf(
    CLASSICAL_DATA_DIR,
    shift,
    medium,
    as_label,
    num_orbitals,
    num_elec_a,
    num_elec_b,
):
    """Load a converged ROHF calculation from a checkpoint file.

    Parameters
    ----------
    CLASSICAL_DATA_DIR : str
        Root directory containing ``chk_files/`` and ``orbitals/``.
    shift : str
        Geometry-shift label (e.g. ``'0.0'``, ``'-0.15'``).
    medium : {'gas', 'solvent'}
        Selects whether PCM solvation is applied.
    as_label : str
        Active-space label, typically ``f'norb_{num_orbitals}_nelec_{na}_{nb}'``.
    num_orbitals : int
        Number of CAS orbitals.
    num_elec_a, num_elec_b : int
        Number of α and β CAS electrons.

    Returns
    -------
    For ``medium == 'gas'``  : ``mf``
    For ``medium == 'solvent'`` : ``(mf, mc, mo_avas)``
    """
    shift_sign = 'minus_' if shift[0] == '-' else ''
    shift_fname_core = (
        shift_sign
        + shift.strip('-').split('.')[0]
        + '_'
        + shift.split('.')[1][::-1].zfill(2)[::-1]
    )
    chkfile_path = f'{CLASSICAL_DATA_DIR}/chk_files/shift_{shift_fname_core}.chk'
    mol = lib.chkfile.load_mol(Path(chkfile_path))
    mf = pyscf.scf.ROHF(mol).density_fit()
    mo_avas = None

    if medium == 'gas':
        mo_coeff = np.load(
            f'{CLASSICAL_DATA_DIR}/orbitals/shift_{shift}/mo_coeff_{as_label}.npy'
        )
        mf.__dict__.update(chkfile.load(chkfile_path, 'scf'))
        mf.converged = True
        mf.mo_coeff = mo_coeff

    if medium == 'solvent':
        mo_coeff = np.load(
            f'{CLASSICAL_DATA_DIR}/orbitals/shift_{shift}/mo_coeff_{as_label}.npy'
        )
        mf = mf.PCM()
        mf.with_solvent.method = 'IEF-PCM'
        mf.with_solvent.eps = 78.3553
        mf.__dict__.update(chkfile.load(chkfile_path, 'scf'))
        mf.converged = True
        mf.mo_coeff = mo_coeff
        mc = mcscf.CASCI(mf, ncas=num_orbitals, nelecas=(num_elec_a, num_elec_b))
        mc = mc.PCM()
        mc.with_solvent.method = mf.with_solvent.method
        mc.with_solvent.eps = mf.with_solvent.eps
        mc.mo_coeff = mo_coeff
        mo_avas = mo_coeff
        return mf, mc, mo_avas

    return mf


def load_converged_scf_and_integrals(
    CLASSICAL_DATA_DIR,
    shift,
    medium,
    as_label,
    num_orbitals,
    num_elec_a,
    num_elec_b,
):
    """Load a converged ROHF calculation **and** active-space integrals
    directly from a PySCF checkpoint (no fcidump needed).

    Returns
    -------
    mf : SCF object (DFROHF or PCM-decorated DFROHF)
    mycas : CASCI object (or PCM-decorated CASCI in solvent)
    h1e, h2e, ecore : np.ndarray, np.ndarray, float
        One- and two-electron integrals in the active space and the core
        (frozen + nuclear) energy offset.
    """
    shift_sign = 'minus_' if shift[0] == '-' else ''
    shift_fname_core = (
        shift_sign
        + shift.strip('-').split('.')[0]
        + '_'
        + shift.split('.')[1][::-1].zfill(2)[::-1]
    )
    chkfile_path = f'{CLASSICAL_DATA_DIR}/chk_files/shift_{shift_fname_core}.chk'
    mol = lib.chkfile.load_mol(Path(chkfile_path))
    mf = pyscf.scf.ROHF(mol).density_fit()

    mo_avas = []
    h1e, h2e, ecore = [], [], []
    mycas = []

    if medium == 'gas':
        mo_coeff = np.load(
            f'{CLASSICAL_DATA_DIR}/orbitals/shift_{shift}/mo_coeff_{as_label}.npy'
        )
        mf.__dict__.update(chkfile.load(chkfile_path, 'scf'))
        mf.converged = True
        mf.mo_coeff = mo_coeff

        mycas = mcscf.CASCI(mf, ncas=num_orbitals, nelecas=(num_elec_a, num_elec_b))
        mycas.mo_coeff = np.load(
            f'{CLASSICAL_DATA_DIR}/orbitals/shift_{shift}/mo_coeff_{as_label}.npy'
        )
        h1e, ecore = mycas.get_h1eff()
        h2e_comp = mycas.get_h2eff()
        h2e = ao2mo.restore(1, h2e_comp, mycas.ncas)

    if medium == 'solvent':
        mo_coeff = np.load(
            f'{CLASSICAL_DATA_DIR}/orbitals/shift_{shift}/mo_coeff_{as_label}.npy'
        )
        mf = mf.PCM()
        mf.with_solvent.method = 'IEF-PCM'
        mf.with_solvent.eps = 78.3553
        mf.__dict__.update(chkfile.load(chkfile_path, 'scf'))
        mf.converged = True
        mf.mo_coeff = mo_coeff

        mycas = mcscf.CASCI(
            mf, ncas=num_orbitals, nelecas=(num_elec_a, num_elec_b)
        ).PCM()
        mycas.with_solvent.method = 'IEF-PCM'
        mycas.with_solvent.eps = 78.3553
        mycas.with_solvent.frozen = True   # <- important, see below
        mycas.mo_coeff = np.load(
            f'{CLASSICAL_DATA_DIR}/orbitals/shift_{shift}/mo_coeff_{as_label}.npy'
        )
        h1e, ecore = mycas.get_h1eff()       # includes PCM reaction-field operator
        h2e_comp = mycas.get_h2eff()         # PCM does NOT touch the 2e part
        h2e = ao2mo.restore(1, h2e_comp, mycas.ncas)

    print(type(mf), type(mycas), type(h1e), type(h2e), type(ecore))
    return mf, mycas, h1e, h2e, ecore


def load_integrals(
    CLASSICAL_DATA_DIR,
    shift,
    spin_number,
    as_label,
    num_orbitals,
):
    """Load active-space integrals (h1e, h2e, ECORE) from an FCIDUMP file."""
    fname = (
        f'{CLASSICAL_DATA_DIR}/active_spaces/shift_{shift}/'
        f'spin_{spin_number}_shift_{shift}_{as_label}.fcidump'
    )
    data = fcidump.read(fname)
    h1e = data['H1']
    h2e_comp = data['H2']
    h2e = ao2mo.restore(1, h2e_comp, num_orbitals)
    ecore = data['ECORE']
    return h1e, h2e, ecore


def load_reference_energy(CLASSICAL_DATA_DIR, as_label, idx):
    """Load the classical CCSD reference total energy at the requested
    geometry-shift index."""
    results = np.load(
        f'{CLASSICAL_DATA_DIR}/active_space_results_classical_{as_label}.npy',
        allow_pickle=True,
    ).item()
    method = 'CCSD'
    ccsd_etot = results['%s_%s' % (as_label, method)]
    return ccsd_etot[idx]


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Loading quantum hardware sampler results
# ══════════════════════════════════════════════════════════════════════════════
def load_hardware_results(
    results_dir: str,
    num_qubits: int,
    shift: float,
    backend_name: str,
):
    """Load a saved Qiskit Runtime sampler PUB result from disk.

    The expected filename layout is::

        {results_dir}/qubits_{num_qubits}_shift_{shift}_{backend_name}.json
    """
    results_file = (
        str(results_dir)
        + f'/qubits_{num_qubits}_shift_{shift}_{backend_name}.json'
    )
    log.info("Loading hardware results from %s", results_file)

    with open(results_file) as f:
        pub_result = json.load(f, cls=RuntimeDecoder)

    num_shots = pub_result.data.meas.num_shots
    num_bits = pub_result.data.meas.num_bits
    log.info(
        "Loaded BitArray: %d shots × %d bits", num_shots, num_bits
    )

    return pub_result


# ══════════════════════════════════════════════════════════════════════════════
# 3.  SQD driver (S-CORE + batched subspace diagonalization)
# ══════════════════════════════════════════════════════════════════════════════
def run_sqd(
    h1e: np.ndarray,
    h2e: np.ndarray,
    bit_array,
    num_orbitals: int,
    num_elec_a: int,
    num_elec_b: int,
    nuclear_repulsion_energy: float,
    # S-CORE options
    max_iterations: int = 10,
    energy_tol: float = 1e-3,
    occupancies_tol: float = 1e-3,
    # Batch / solver options
    num_batches: int = 3,
    samples_per_batch: int = 300,
    symmetrize_spin: bool = True,
    carryover_threshold: float = 1e-4,
    max_cycle: int = 200,
    spin_sq: float = 0.0,
    seed: int = 12345,
):
    """Run the full SQD self-consistent configuration recovery + diagonalization
    loop and return the best result, the per-iteration history, and wall time."""

    # Configure the SCI solver with the target ⟨S²⟩ and max Davidson cycles
    sci_solver = partial(
        solve_sci_batch,
        spin_sq=spin_sq,
        max_cycle=max_cycle,
    )

    # Accumulator for intermediate results from each S-CORE iteration
    result_history: list[list[SCIResult]] = []
    t_start = time.time()

    def callback(results: list[SCIResult]):
        """
        Called at the end of each S-CORE iteration with the batch results.
        Logs energies and subspace dimensions for monitoring convergence.
        """
        result_history.append(results)
        iteration = len(result_history)
        log.info("─── S-CORE iteration %d ───", iteration)
        for i, res in enumerate(results):
            e_total = res.energy + nuclear_repulsion_energy
            dim = np.prod(res.sci_state.amplitudes.shape)
            elapsed = (time.time() - t_start) / 60.0
            log.info(
                "  Batch %d  │  E_total = %.10f Ha  │  dim = %d  │  %.1f min",
                i, e_total, dim, elapsed,
            )

    # ── Main SQD call ──────────────────────────────────────────────────────
    log.info("Starting SQD post-processing (%d orbitals, %d↑ + %d↓ electrons)",
             num_orbitals, num_elec_a, num_elec_b)
    log.info("  max_iterations=%d, num_batches=%d, samples_per_batch=%d",
             max_iterations, num_batches, samples_per_batch)

    result = diagonalize_fermionic_hamiltonian(
        h1e,
        h2e,
        bit_array,
        samples_per_batch=samples_per_batch,
        norb=num_orbitals,
        nelec=(num_elec_a, num_elec_b),
        num_batches=num_batches,
        energy_tol=energy_tol,
        occupancies_tol=occupancies_tol,
        max_iterations=max_iterations,
        sci_solver=sci_solver,
        symmetrize_spin=symmetrize_spin,
        carryover_threshold=carryover_threshold,
        callback=callback,
        seed=seed,
    )

    wall_time = time.time() - t_start
    best_energy = result.energy + nuclear_repulsion_energy
    log.info("SQD complete in %.1f min  │  Best E_total = %.10f Ha",
             wall_time / 60.0, best_energy)

    return result, result_history, wall_time


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Analysis & plotting
# ══════════════════════════════════════════════════════════════════════════════
def analyze_convergence(
    result_history: list[list[SCIResult]],
    nuclear_repulsion_energy: float,
    reference_energy: float = None,
):
    """
    Extract per-iteration best energies and compute energy errors
    relative to a classical reference (if provided).

    Returns
    -------
    iterations : list[int]
        0-indexed iteration numbers.
    best_energies : list[float]
        Best (min over batches) total energy at each iteration.
    energy_errors : list[float] or None
        |E_SQD - E_ref| at each iteration.  None if no reference given.
    """
    iterations = list(range(len(result_history)))
    best_energies = [
        min(batch, key=lambda r: r.energy).energy + nuclear_repulsion_energy
        for batch in result_history
    ]

    energy_errors = None
    if reference_energy is not None:
        energy_errors = [abs(e - reference_energy) for e in best_energies]

    return iterations, best_energies, energy_errors


def plot_results(
    result: SCIResult,
    result_history: list[list[SCIResult]],
    nuclear_repulsion_energy: float,
    reference_energy: float = None,
    save_path: str = None,
):
    """
    Produce the standard two-panel SQD diagnostic figure:
        Left  — energy error vs. S-CORE iteration (log scale)
        Right — average spatial orbital occupancy after final iteration

    Parameters
    ----------
    result : SCIResult
        Final (best) SQD result carrying orbital_occupancies.
    result_history : list[list[SCIResult]]
        Full iteration history.
    nuclear_repulsion_energy : float
        Constant offset for total energies.
    reference_energy : float, optional
        Classical reference energy for error computation.
    save_path : str, optional
        If provided, save figure to this path instead of displaying.
    """
    iters, best_e, e_errors = analyze_convergence(
        result_history, nuclear_repulsion_energy, reference_energy
    )

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # ── Left panel: energy convergence ─────────────────────────────────────
    if e_errors is not None:
        axs[0].plot(iters, e_errors, marker="o", linewidth=2, label="Energy error")
        axs[0].set_yscale("log")
        axs[0].set_ylim(bottom=1e-5)

        # Chemical accuracy line (1 kcal/mol ≈ 1.6 mHa)
        chem_accuracy_ha = 1.6e-3
        axs[0].axhline(
            y=chem_accuracy_ha,
            color="#BF5700",
            linestyle="--",
            linewidth=1.5,
            label="Chemical accuracy (1.6 mHa)",
        )
        axs[0].set_ylabel("| E_SQD − E_ref |  (Ha)", fontsize=12)
    else:
        # No reference — just plot raw energies
        axs[0].plot(iters, best_e, marker="o", linewidth=2, label="Best E_total")
        axs[0].set_ylabel("Total Energy (Ha)", fontsize=12)

    axs[0].set_xticks(iters)
    axs[0].set_xlabel("S-CORE Iteration", fontsize=12)
    axs[0].set_title("Ground-State Energy Convergence")
    axs[0].legend(fontsize=10)
    axs[0].grid(True, alpha=0.3)

    # ── Right panel: orbital occupancy ─────────────────────────────────────
    # result.orbital_occupancies has shape (2, num_orbitals) for (alpha, beta)
    occ = np.sum(result.orbital_occupancies, axis=0)  # sum over spin
    orb_idx = np.arange(len(occ))

    axs[1].bar(orb_idx, occ, width=0.7, color="#4C72B0", edgecolor="white")
    axs[1].set_xticks(orb_idx)
    axs[1].set_xlabel("Spatial Orbital Index", fontsize=12)
    axs[1].set_ylabel("Avg. Occupancy (α + β)", fontsize=12)
    axs[1].set_title("Orbital Occupancies (Final Iteration)")
    axs[1].set_ylim(0, 2.1)
    axs[1].grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        log.info("Figure saved to %s", save_path)
    else:
        plt.show()


def print_summary(
    result: SCIResult,
    result_history: list[list[SCIResult]],
    nuclear_repulsion_energy: float,
    reference_energy: float = None,
    wall_time: float = None,
    *,
    num_elec_a: int,
    num_elec_b: int,
    num_orbitals: int,
    num_batches: int,
    samples_per_batch: int,
):
    """Print a concise summary table to stdout.

    The active-space dimensions and batching parameters are passed in
    explicitly (rather than read from globals as in the original notebook)
    so this function is fully reusable.
    """
    iters, best_e, e_errors = analyze_convergence(
        result_history, nuclear_repulsion_energy, reference_energy
    )

    print("\n" + "=" * 72)
    print("  SQD POST-PROCESSING SUMMARY")
    print("=" * 72)
    print(f"  Active space       : ({num_elec_a + num_elec_b}e, {num_orbitals}o)")
    print(f"  S-CORE iterations  : {len(result_history)}")
    print(f"  Batches / iteration: {num_batches}")
    print(f"  Samples per batch  : {samples_per_batch}")

    if wall_time is not None:
        print(f"  Wall-clock time    : {wall_time / 60:.1f} min")

    print("-" * 72)
    print(f"  {'Iter':>4s}  {'Best E_total (Ha)':>22s}", end="")
    if e_errors is not None:
        print(f"  {'Error (Ha)':>14s}  {'Error (kcal/mol)':>16s}", end="")
    print()
    print("-" * 72)

    for i, e in enumerate(best_e):
        line = f"  {i:4d}  {e:22.10f}"
        if e_errors is not None:
            err_ha = e_errors[i]
            err_kcal = err_ha * 627.509  # 1 Ha = 627.509 kcal/mol
            line += f"  {err_ha:14.6e}  {err_kcal:16.4f}"
        print(line)

    print("-" * 72)
    final_e = best_e[-1]
    print(f"  Final best energy  : {final_e:.10f} Ha")

    if reference_energy is not None:
        final_err = e_errors[-1]
        print(f"  Reference energy   : {reference_energy:.10f} Ha")
        print(f"  Final error        : {final_err:.6e} Ha"
              f"  ({final_err * 627.509:.4f} kcal/mol)")
        if final_err < 1.6e-3:
            print("  ✓ Within chemical accuracy (1 kcal/mol ≈ 1.6 mHa)")
        else:
            print("  ✗ Not yet at chemical accuracy — consider increasing"
                  " samples_per_batch or max_iterations")

    print("=" * 72 + "\n")
