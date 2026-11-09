"""
================================================================================
utils_sqd_post_process_solvent.py
   — Helper functions for SQD Post-Processing in the IEF-PCM solvent medium
================================================================================

This module collects all helper functions used by the solvent SQD
post-processing pipeline (charge-conjugate reaction-field embedding):

    Loading classical inputs
    ------------------------
    - load_converged_scf
    - load_converged_scf_and_integrals
    - load_integrals
    - load_reference_energy

    Loading quantum hardware data
    -----------------------------
    - load_hardware_results

    Density-matrix utilities
    ------------------------
    - update_rdm                 (CAS-MO 1-RDM  →  full AO basis)

    SQD drivers
    -----------
    - run_sqd_inner              (one S-CORE + diagonalization loop)
    - run_sqd_solvent            (outer SCRF loop wrapping run_sqd_inner)

    Analysis & plotting
    -------------------
    - plot_scrf_convergence
    - plot_inner_convergence
    - print_summary

All functions are self-contained: any quantity previously read from a
notebook-level global (``as_label``, ``num_orbitals``, ``num_elec_a/b``,
``spin_number``, ``RESULTS_DIR``, ``max_iterations``, ``num_batches``,
``samples_per_batch``) is now an explicit argument.

Physics summary
---------------
Inside ``run_sqd_solvent`` we iterate a self-consistent reaction-field
(SCRF) loop.  At iteration k we

    1.  Set h_core ← h_core^(gas) + V_solvent^(k-1).
    2.  Recompute CAS integrals (h1e_cas, h2e_cas, ECORE) with the
        current solvent potential embedded in the one-body part.
    3.  Run an inner S-CORE + batched subspace diagonalization to get
        |ψ_k⟩, E_active^(k), and the spin-summed 1-RDM ρ_CAS.
    4.  Back-rotate ρ_CAS to the full AO basis ρ_AO and feed it into the
        IEF-PCM solver to obtain the new solvent energy E_solv^(k) and
        potential V_solvent^(k).
    5.  Apply the half-counting correction
            E_total^(k) = E_active^(k) + ECORE + E_solv^(k-1)
                          − Tr[V_solvent^(k-1) · ρ_AO]
        (the first iteration omits the solvent terms because no V is
        available yet).

Reference:
    Robledo-Moreno et al., "Chemistry Beyond Exact Solutions on a
    Quantum-Centric Supercomputer", arXiv 2405.05068 (2024);
    Tomasi, Mennucci & Cammi, Chem. Rev. 105, 2999 (2005)  for IEF-PCM.
================================================================================
"""

import os
import json
import time
import logging
from functools import partial, reduce
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
# Module logger (the caller is responsible for logging.basicConfig)
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
        Selects whether IEF-PCM solvation is applied.
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
    directly from a PySCF checkpoint (no FCIDUMP needed).

    NOTE: in the solvent branch the PCM model is **not** frozen here
    (``mycas.with_solvent.frozen`` is left to its PySCF default).  The SCRF
    loop in :func:`run_sqd_solvent` toggles the reaction-field potential
    explicitly, so freezing it at this stage would lock the calculation to
    the gas-phase orbital relaxation.

    Returns
    -------
    mf : SCF object (DFROHF or PCM-decorated DFROHF)
    mycas : CASCI object (or PCM-decorated CASCI in solvent)
    h1e, h2e, ecore : np.ndarray, np.ndarray, float
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
        mycas.with_solvent.method = mf.with_solvent.method
        mycas.with_solvent.eps = mf.with_solvent.eps
        # mycas.with_solvent.frozen = True
        # ↑ intentionally NOT frozen — the SCRF loop in run_sqd_solvent
        #   updates V_solvent each iteration.
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
# 3.  Density-matrix utility
# ══════════════════════════════════════════════════════════════════════════════
def update_rdm(casci_object, dmas):
    """Build the full-AO 1-particle density matrix from the CAS-MO 1-RDM.

    ρ_AO = ρ_core + C_cas · ρ_CAS · C_caŝ†

    where ρ_core = 2 · C_core · C_corê† (closed-shell core) and the active
    block ρ_CAS is the spin-summed CAS 1-RDM in the CAS-MO basis.

    Parameters
    ----------
    casci_object : pyscf.mcscf.CASCI (or PCM-decorated variant)
        Provides ``mo_coeff``, ``ncore``, ``ncas``.
    dmas : np.ndarray, shape (ncas, ncas)
        Spin-summed 1-RDM in the CAS-MO basis.

    Returns
    -------
    rho_ao : np.ndarray, shape (nao, nao)
        Total 1-RDM in the AO basis.
    """
    mo_coeff = casci_object.mo_coeff
    ncore = casci_object.ncore
    ncas = casci_object.ncas

    mocore = mo_coeff[:, :ncore]
    mocas = mo_coeff[:, ncore:ncore + ncas]

    # Core contribution: doubly occupied orbitals
    dm_core = np.dot(mocore, mocore.conj().T) * 2

    # Active-space contribution: back-rotate rdm1 from CAS MO → AO
    dm_cas = reduce(np.dot, (mocas, dmas, mocas.conj().T))

    rho_ao = dm_core + dm_cas
    return rho_ao


# ══════════════════════════════════════════════════════════════════════════════
# 4.  SQD drivers
# ══════════════════════════════════════════════════════════════════════════════
def run_sqd_inner(
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
    """One inner S-CORE + batched subspace-diagonalization pass.

    This is invoked once per outer SCRF iteration in
    :func:`run_sqd_solvent` and is the solvent counterpart of the
    standalone gas-phase ``run_sqd``.
    """

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
        """Logs energies and subspace dimensions for each S-CORE iteration."""
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


def run_sqd_solvent(
    mc,
    bit_array,
    num_orbitals: int,
    num_elec_a: int,
    num_elec_b: int,
    n_scrf_iter: int = 5,
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
    """Outer self-consistent reaction-field (SCRF) loop wrapping
    :func:`run_sqd_inner`.

    Iterates: SQD → ρ_AO → IEF-PCM → V_solvent → updated h_core,
    until ``n_scrf_iter`` iterations have completed.

    Returns
    -------
    scrf_data : list[dict]
        One record per SCRF iteration with keys
        ``scrf_iter, e_active, ecore, e_total, e_solvent,
          e_double_count, rho_ao, rdm1_cas, inner_history, wall_time``.
    final_result : SCIResult
        The last inner SQD result (best batch of the last iteration).
    """
    # ── Validate that the CASCI object has solvent attached ────────────────
    if not hasattr(mc, "with_solvent"):
        raise AttributeError(
            "The CASCI object `mc` does not have a solvent model attached.\n"
            "Ensure you created it with e.g.:\n"
            "    mc = mcscf.CASCI(mf, ncas, nelecas)\n"
            "    mc = solvent.ddCOSMO(mc)  # or mc.PCM()\n"
        )

    # ── Initialise solvent state ──────────────────────────────────────────
    with_solvent_e = None   # scalar solvent polarisation energy
    with_solvent_v = None   # Fock-like potential matrix (nao, nao)

    scrf_data = []
    total_wall = 0.0

    for k in range(n_scrf_iter):
        log.info("=" * 70)
        log.info("SCRF ITERATION %d / %d", k + 1, n_scrf_iter)
        log.info("=" * 70)

        # ── Step 1: update core Hamiltonian ───────────────────────────────
        if with_solvent_v is not None:
            # Subsequent iterations: augment h_core with solvent potential.
            # Use a default-argument closure to capture the current value of
            # with_solvent_v (avoids late-binding issues with lambda).
            _v = with_solvent_v.copy()
            mc.get_hcore = lambda *args, _vv=_v: mc._scf.get_hcore() + _vv
            log.info("  h_core updated: gas-phase + V_solvent")
        else:
            # First iteration: gas-phase core Hamiltonian only.
            mc.get_hcore = lambda *args: mc._scf.get_hcore()
            log.info("  h_core: gas-phase (no solvent potential yet)")

        # ── Step 2: recompute CAS integrals ───────────────────────────────
        h1e_cas, ecore = mc.get_h1eff()
        h2e_cas = ao2mo.restore(1, mc.get_h2eff(), num_orbitals)
        log.info("  ecore = %.10f Ha", ecore)

        # ── Step 3: inner S-CORE + batched diagonalisation ────────────────
        result, inner_history, wt = run_sqd_inner(
            h1e=h1e_cas,
            h2e=h2e_cas,
            bit_array=bit_array,
            num_orbitals=num_orbitals,
            num_elec_a=num_elec_a,
            num_elec_b=num_elec_b,
            nuclear_repulsion_energy=ecore,
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
        total_wall += wt
        e_active = result.energy
        rdm1_cas = result.rdm1     # spin-summed 1-RDM in CAS MO basis

        # ── Step 4: back-rotate rdm1 to full AO basis ────────────────────
        rho_ao = update_rdm(mc, rdm1_cas).copy()

        # ── Step 5: energy correction ─────────────────────────────────────
        if with_solvent_e is not None:
            # Double-counting correction: the solvent potential V_solvent is
            # already embedded inside h1e_cas via get_h1eff().  The
            # expectation value ⟨V_solvent⟩ = Tr(V_solvent · ρ_AO) has thus
            # been counted once in e_active.  The correct total free energy
            # (Eq. 3) requires only ½ of this term, so we subtract the
            # overcounted half: E = e + ecore + E_solv − Tr(V·ρ).
            e_dup = np.einsum("ij,ji->", with_solvent_v, rho_ao)
            e_total = e_active + ecore + with_solvent_e - e_dup
            log.info("  E_active = %.10f", e_active)
            log.info("  E_solv   = %.10f", with_solvent_e)
            log.info("  E_dup    = %.10f", e_dup)
        else:
            # First iteration: no solvent correction yet.
            e_dup = None
            e_total = e_active + ecore

        log.info("  E_total  = %.10f Ha", e_total)

        # ── Step 6: compute IEF-PCM solvent response ──────────────────────
        with_solvent_e, with_solvent_v = mc.with_solvent._get_vind(rho_ao)
        log.info(
            "  Solvent response computed: E_solv = %.10f", with_solvent_e
        )

        # ── Record this iteration ─────────────────────────────────────────
        scrf_data.append({
            "scrf_iter":      k,
            "e_active":       float(e_active),
            "ecore":          float(ecore),
            "e_total":        float(e_total),
            "e_solvent":      float(with_solvent_e) if with_solvent_e is not None else None,
            "e_double_count": float(e_dup) if e_dup is not None else None,
            "rho_ao":         rho_ao,
            "rdm1_cas":       rdm1_cas,
            "inner_history":  inner_history,
            "wall_time":      wt,
        })

    log.info("=" * 70)
    log.info(
        "SCRF loop complete in %.1f min  |  Final E_total = %.10f Ha",
        total_wall / 60.0, scrf_data[-1]["e_total"],
    )
    log.info("=" * 70)

    return scrf_data, result


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Analysis & plotting
# ══════════════════════════════════════════════════════════════════════════════
def plot_scrf_convergence(
    scrf_data: list[dict],
    reference_energy: float = None,
    save_path: str = None,
):
    """
    Three-panel diagnostic figure for the SQD IEF-PCM workflow:
        Left   — Total energy vs. SCRF iteration
        Center — Energy error vs. SCRF iteration (log scale, if reference given)
        Right  — Solvent energy contribution vs. SCRF iteration
    """
    iters = [d["scrf_iter"] for d in scrf_data]
    e_totals = [d["e_total"] for d in scrf_data]
    e_solvents = [d["e_solvent"] for d in scrf_data]

    n_panels = 3 if reference_energy is not None else 2
    fig, axs = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))

    # ── Panel 1: total energy convergence ─────────────────────────────────
    ax = axs[0]
    ax.plot(iters, e_totals, "o-", linewidth=2, color="#1f77b4", label="E_total")
    if reference_energy is not None:
        ax.axhline(
            y=reference_energy, color="grey", linestyle="--",
            linewidth=1.5, label="CASCI IEF-PCM ref",
        )
    ax.set_xlabel("SCRF Iteration", fontsize=12)
    ax.set_ylabel("Total Energy (Ha)", fontsize=12)
    ax.set_title("SQD IEF-PCM Energy Convergence")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(iters)

    # ── Panel 2: energy error (if reference available) ────────────────────
    panel_idx = 1
    if reference_energy is not None:
        ax = axs[panel_idx]
        errors = [abs(e - reference_energy) for e in e_totals]
        ax.plot(iters, errors, "s-", linewidth=2, color="#d62728", label="Error")
        ax.set_yscale("log")
        chem_acc = 1.6e-3
        ax.axhline(
            y=chem_acc, color="#BF5700", linestyle="--",
            linewidth=1.5, label="Chemical accuracy (1.6 mHa)",
        )
        ax.set_xlabel("SCRF Iteration", fontsize=12)
        ax.set_ylabel("|E_SQD - E_ref| (Ha)", fontsize=12)
        ax.set_title("Energy Error vs. SCRF Iteration")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(iters)
        panel_idx += 1

    # ── Final panel: solvent energy ───────────────────────────────────────
    ax = axs[panel_idx]
    # First iteration may have None for e_solvent; replace with 0 for plotting
    e_solv_plot = [e if e is not None else 0.0 for e in e_solvents]
    ax.plot(iters, e_solv_plot, "^-", linewidth=2, color="#2ca02c", label="E_solvent")
    ax.set_xlabel("SCRF Iteration", fontsize=12)
    ax.set_ylabel("Solvent Energy (Ha)", fontsize=12)
    ax.set_title("IEF-PCM Solvent Energy")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(iters)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        log.info("Figure saved to %s", save_path)
    else:
        plt.show()


def plot_inner_convergence(
    scrf_data: list[dict],
    ecore_per_iter: list[float] = None,
    save_path: str = None,
):
    """
    Plot the S-CORE inner-loop convergence for each SCRF iteration.

    Each SCRF iteration's inner S-CORE history is plotted as a separate
    line showing the best batch energy vs. inner iteration number.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for d in scrf_data:
        k = d["scrf_iter"]
        ecore = d["ecore"]
        inner_hist = d["inner_history"]

        best_energies = []
        for step_results in inner_hist:
            best_e = min(r.energy + ecore for r in step_results)
            best_energies.append(best_e)

        inner_iters = list(range(len(best_energies)))
        ax.plot(
            inner_iters, best_energies, "o-",
            label=f"SCRF iter {k}", alpha=0.8,
        )

    ax.set_xlabel("S-CORE Inner Iteration", fontsize=12)
    ax.set_ylabel("Best E_total (Ha)", fontsize=12)
    ax.set_title("Inner S-CORE Convergence per SCRF Iteration")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        log.info("Figure saved to %s", save_path)
    else:
        plt.show()


def print_summary(
    scrf_data: list[dict],
    reference_energy: float = None,
    *,
    num_elec_a: int,
    num_elec_b: int,
    num_orbitals: int,
    max_iterations: int,
    num_batches: int,
    samples_per_batch: int,
):
    """Print a concise summary table of the SCRF iterations.

    The active-space dimensions and SQD batching parameters are passed in
    explicitly (rather than read from globals as in the original notebook)
    so this function is fully reusable.
    """
    print("\n" + "=" * 80)
    print("  SQD IEF-PCM POST-PROCESSING SUMMARY")
    print("=" * 80)
    print(f"  Active space       : ({num_elec_a + num_elec_b}e, {num_orbitals}o)")
    print(f"  SCRF iterations    : {len(scrf_data)}")
    print(f"  S-CORE iters/cycle : {max_iterations} (max)")
    print(f"  Batches / S-CORE   : {num_batches}")
    print(f"  Samples per batch  : {samples_per_batch}")

    total_wall = sum(d["wall_time"] for d in scrf_data)
    print(f"  Total wall time    : {total_wall / 60:.1f} min")

    print("-" * 80)
    header = (
        f"  {'SCRF':>4s}  {'E_active (Ha)':>18s}  {'ecore (Ha)':>16s}"
        f"  {'E_total (Ha)':>18s}  {'E_solvent (Ha)':>16s}"
    )
    if reference_energy is not None:
        header += f"  {'Error (mHa)':>12s}"
    print(header)
    print("-" * 80)

    for d in scrf_data:
        e_solv_str = (
            f"{d['e_solvent']:16.8f}" if d["e_solvent"] is not None
            else "        N/A     "
        )
        line = (
            f"  {d['scrf_iter']:4d}"
            f"  {d['e_active']:18.10f}"
            f"  {d['ecore']:16.10f}"
            f"  {d['e_total']:18.10f}"
            f"  {e_solv_str}"
        )
        if reference_energy is not None:
            err_mha = abs(d["e_total"] - reference_energy) * 1000
            line += f"  {err_mha:12.4f}"
        print(line)

    print("-" * 80)
    final_e = scrf_data[-1]["e_total"]
    print(f"  Final E_total      : {final_e:.10f} Ha")

    if reference_energy is not None:
        final_err = abs(final_e - reference_energy)
        print(f"  Reference energy   : {reference_energy:.10f} Ha")
        print(f"  Final error        : {final_err:.6e} Ha"
              f"  ({final_err * 627.509:.4f} kcal/mol)")
        if final_err < 1.6e-3:
            print("  [OK] Within chemical accuracy (1 kcal/mol ~ 1.6 mHa)")
        else:
            print("  [!!] Not at chemical accuracy — consider more SCRF iters"
                  " or larger samples_per_batch")

    # Solvation free energy estimate
    if len(scrf_data) >= 2:
        # Compare final solvated energy to first (gas-phase-like) energy
        delta_g = scrf_data[-1]["e_total"] - scrf_data[0]["e_total"]
        print(f"\n  Approximate solvation correction: {delta_g:.6f} Ha"
              f"  ({delta_g * 627.509:.4f} kcal/mol)")
        print("  (NOTE: for a rigorous G_solv, compare to a separate gas-phase"
              " SQD calculation.)")

    print("=" * 80 + "\n")
