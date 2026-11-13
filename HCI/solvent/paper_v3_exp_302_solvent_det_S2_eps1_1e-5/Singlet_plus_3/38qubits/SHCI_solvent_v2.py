# %%
"""
SHCI (Dice) reference calculation for the Co(5H2O)CO2 complex in implicit solvent
(IEF-PCM, water, eps = 78.3553), with a self-consistent reaction field (SCRF) loop,
including the spin purity <S^2> of the SHCI wavefunction in every SCRF iteration.

Usage (run from notebooks/analysis/SHCI/, see run_SHCI.sh / run_SHCI_concurrent.sh):
    python SHCI_solvent_v2.py <shift> <spin_number> <n_so> [<charge>]

    shift        CO2 displacement as in the .xyz file names, e.g. '-0.15', '0.0', '1.2'
    spin_number  2S = n_alpha - n_beta: 0 Singlet, 1 Doublet, 3 Quartet, 4 Quintet
    n_so         number of active spin-orbitals (= qubits), n_orb = n_so // 2
    charge       total charge, optional, default 2 (charge 3 -> Singlet/Quintet)

What the script does
--------------------
1. Loads the converged IEF-PCM ROHF (density-fitted) and AVAS active-space orbitals from
   data/interim/.../MF/ROHF_stab_analysis_5/solvent/charge{charge}_spin{spin}/.
2. SCRF loop, n_scrf_iter iterations k = 0 .. n_scrf_iter-1. In iteration k:
   a. h_core = gas-phase h_core (+ V_solvent from iteration k-1 for k > 0), active-space
      integrals rebuilt via CASCI.get_h1eff / get_h2eff (ecore passed to Dice is 0, the
      CAS ecore is added afterwards).
   b. PASS 1 (production): variational HCI + deterministic PT with Dice (run_SHCI).
      Its energy enters e_active / e_total of iteration k. Unchanged by the <S^2> addition.
   c. The variational spin-resolved 1-RDM (spin1RDM.0.0.txt) is read right after pass 1.
      Dice writes the 1-RDMs inside the variational step and never overwrites them in PT,
      so the density used for the reaction field is the variational one (as before).
   d. PASS 2 (<S^2>): Dice fullrestart from the pass-1 variational space with PT switched
      off, to regenerate the variational spin-free 2-RDM, from which <S^2> is evaluated
      (details and reasons in run_SHCI).
   e. Back-rotation of the 1-RDM to the AO basis, double-counting correction, new IEF-PCM
      response (E_solv, V_solvent) for iteration k+1.
3. Saves the SCRF history (pickle + JSON summary), prints a summary table and a convergence
   plot. The reported solvent energy is e_total of the LAST SCRF iteration, and the
   reported <S^2> is the one of the LAST SCRF iteration (same wavefunction).

Outputs (in data/interim/.../SHCI/{basis}/AVAS/solvent/{exp_name}/{Spin}_plus_{charge}/{n_so}qubits/)
    HCI_shift_{shift}.log                 log of the run, APPENDED on every rerun. Contains per
                                          SCRF iteration "S^2 = ...", "S(S+1) exact = ...",
                                          "S^2 deviation = ...", "E_var ... = ..." lines.
                                          C1.ipynb parses the LAST "S^2 = " line, i.e. the final
                                          SCRF iteration of the latest run.
    HCI_shift_{shift}_results_raw.pkl     list (one dict per SCRF iteration) with e_active, ecore,
                                          e_total, e_solvent, e_double_count, rho_ao, rdm1_cas,
                                          wall_time and the <S^2> keys (s2, s2_exact, e_var_active,
                                          e_var_active_rdm, rdm_check_ok, s2_wall_time)
    HCI_shift_{shift}_results_summary.json  the scalar part of the above, per SCRF iteration
    HCI_shift_{shift}_convergence.png     SCRF convergence plot
    shift_{shift}_results_k_{k}.pkl       per SCRF iteration: SHCI energy, solver metadata of
                                          pass 1, runtime, <S^2> keys (see run_SHCI)
    shift_{shift}_results_rawk_{k}.pkl    per SCRF iteration: raw return value of pass-1 kernel()
    SHCI_input/output_{shift}.dat         Dice input/output of pass 1, LAST SCRF iteration only
                                          (overwritten every iteration) + appended <S^2> summary
    SHCI_input/output_{shift}_S2.dat      Dice input/output of pass 2, last SCRF iteration
    SHCI_FCIDUMP_{shift}                  deleted at the end of every SCRF iteration
Scratch dir (notebooks/analysis/SHCI/scratch/solvent/{exp_name}/.../{shift}/): after every SCRF
iteration only the variational RDM text files + shci.e of that iteration are kept (see run_SHCI).

History
-------
- exp_name 'paper_v3_exp_301_solvent_det' / '..._det_v2': energies only (no <S^2>), charge +2
  hard-coded (charge +3 data in '..._det' were produced with a charge-3 copy of this script).
- exp_name 'paper_v3_exp_301_solvent_det_S2': same Dice and SCRF settings, plus <S^2>,
  optional charge argument; added for the JCTC review comment on spin purity (SI S1.7).
  The gas-phase counterpart is SHCI_gas_v3_new_handlers.py ('..._gas_new_loader_S2').
- exp_name 'paper_v3_exp_302_solvent_det_S2_eps1_1e-4': as exp_301_..._S2, but epsilon1
  (sweep_epsilon) = 1e-4 instead of 1e-3 (SHCI_EPS1=1e-4); everything else unchanged.
"""
import os

# ----- threads -----
# Must be set BEFORE numpy/pyscf are imported: BLAS/OpenMP read these variables only once,
# at library load. The Dice subprocess inherits them (Dice is built with OpenMP).
# Values already exported by the caller win (run_SHCI_concurrent.sh exports
# OMP_NUM_THREADS=<threads per job>); otherwise all cores are used (serial run_SHCI.sh).
n_threads = int(os.environ.get("OMP_NUM_THREADS", os.cpu_count()))
for var in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ.setdefault(var, str(n_threads))

import json
import time
import logging
from functools import reduce
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import pyscf
from pyscf import gto, scf, mp, mcscf, cc, ci, tools, ao2mo, lib
from pyscf.scf import chkfile
from pyscf import lib
from pyscf.tools import fcidump
from pyscf.shciscf import shci


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
        f'{CLASSICAL_DATA_DIR}/active_spaces/'
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
    method = 'CCSD(T)'
    ccsd_etot = results['%s_%s' % (as_label, method)]
    return ccsd_etot[idx]


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


def run_SHCI(mol, res, shift_fname_core, res_str, scratch_dir, spin_number):
    mch = shci.SHCI(mol)
    mch.mpiprefix = '' #'mpirun -np 5'
    # Launch Dice through dice_start_guard.sh instead of the bare binary: Dice names its shared-
    # memory segments only by time(NULL) in seconds, so two Dice processes started in the same
    # second (concurrent jobs, run_SHCI_concurrent.sh) overwrite each other's data (observed:
    # segfault, exit 139, Quintet +3 50 qubits; silent corruption is also possible). The guard
    # lets at most one Dice start at a time machine-wide; see dice_start_guard.sh.
    mch.executable = f"bash {Path('./notebooks/analysis/SHCI/dice_start_guard.sh').resolve()}"

    # variational HCISCI run
    mch.dE = 1.0e-8 # The energy tolerance used to control convergence during the variational HCI algorithm.
    mch.davidsonTol = 5.0e-5
    mch.maxIter = 20

    # mch.epsilon1 = 1e-3
    mch.sweep_iter = [20] # [5, 20] - from 0-4th first sweep epsilon, then up to 19th second
    mch.sweep_epsilon = [eps1]  # module level: 1e-3 published (exp_301), 1e-4 in exp_302 (SHCI_EPS1) #[ 1e-4, 3e-6]

    # PT run 
    mch.stochastic = False
    mch.epsilon2 = 1.0e-5
    if mch.stochastic:
        mch.sampleN = 200 #TODO 20000 # (Integer) Number of times the set of determinants outside 
                            # the variational space is sampled in a given stochastic or 
                            # semistochastic iteration.
        mch.nPTiter = 100 #TODO 10 # The maximum number of iterations for the stochastic/semi-stochastic PT energy correction. 0 turns off.
        mch.epsilon2Large = 1000 #TODO 10000 # controls what portion of the perturbative
                #correction is calculated deterministically (when calculating the 
                #correction) using semi-stochastic HCI.
        mch.targetError = 1.0e-5 #TODO 1.0e-5 #The tolerance for stochastic/semi-stochastic PT correction to the HCI energy.

    # DoRDM: Dice writes the spin-free (spatial) 2-RDM to {scratch}/spatialRDM.0.0.txt.
    # CAUTION: with deterministic PT (stochastic=False) Dice OVERWRITES this file at the end
    # of the PT step with a PT-response RDM, i.e. after pass 1 it does NOT hold the
    # variational 2-RDM -> pass 2 below. The 1-RDM files (spin1RDM / spatial1RDM) are
    # written in the variational step and are not touched by PT.
    mch.DoRDM = True

    mch.integralFile = (res_str+f"SHCI_FCIDUMP_{shift_fname_core}")
    mch.configFile =  (res_str+f"SHCI_input_{shift_fname_core}.dat")
    mch.outputFile = (res_str+f"SHCI_output_{shift_fname_core}.dat")
    mch.scratchDirectory += scratch_dir

    mch.printbestdeterminants = True
    nelec = ((res['NELEC']+spin_number)//2, (res['NELEC']-spin_number)//2)

    # =========================================================================
    # PASS 1: production SHCI run (variational HCI + deterministic PT)
    # -> SHCI_res[0] is the ACTIVE-SPACE energy (ECORE = 0 here) of this SCRF iteration.
    # =========================================================================
    log.info("[PASS 1] Production SHCI run: variational HCI + deterministic PT")
    start_tmie = time.time()
    SHCI_res = mch.kernel(res['H1'],
                        res['H2'],
                        norb=res['NORB'],
                        nelec=nelec,
                        fciRestart=False,
                        ecore=res['ECORE'])

    end_time = time.time()
    SHCI_runtime = end_time - start_tmie

    # Variational spin-resolved 1-RDM (alpha, beta) for the reaction field. Read BEFORE
    # pass 2, exactly as in the energy-only version of this script, so the SCRF density
    # (and therefore all energies) are unchanged by the <S^2> addition.
    rdm1_cas = mch.make_rdm1s()

    # Snapshot the solver settings NOW: pass 2 changes stochastic/nPTiter/file names,
    # and the stored metadata must describe the production run (pass 1).
    mch.stdout = mol.output
    SHCI_metadata = dict(vars(mch))
    main_output_file = mch.outputFile

    # =========================================================================
    # PASS 2: <S^2> of the variational SHCI wavefunction (same method as in
    # SHCI_gas_v3_new_handlers.py, see the comments there for the validation)
    # =========================================================================
    # - pyscf.shciscf.shci.SHCI.spin_square() does NOT compute anything: it returns the
    #   nominal S(S+1) from (n_alpha - n_beta)/2, so it cannot detect spin contamination.
    # - <S^2> from the spin-free 2-RDM (PySCF chemist convention,
    #   dm2[p,q,r,s] = sum_{sigma,tau} <a^+_{p sigma} a^+_{r tau} a_{s tau} a_{q sigma}>):
    #       <S^2> = -N(N-4)/4 - 1/2 * sum_pq dm2[p,q,q,p],   N = active electrons
    #   (closed-shell core contributes nothing; validated against pyscf FCI spin_square).
    # - After pass 1 the 2-RDM on disk is a PT-response RDM (see DoRDM above), so Dice is
    #   restarted from the converged determinant space of pass 1 ("fullrestart", reads
    #   {scratch}/0-variational.bkp) with stochastic=True, nPTiter=0: PT is switched off and
    #   Dice skips the PT-RDM step ("We cannot perform PT RDM with stochastic PT. Disabling
    #   RDM."), so the variational 2-RDM stays on disk. Cost: ~2 Davidson iterations.
    # - The Hamiltonian of pass 2 is the same as in pass 1 (same SCRF iteration, same
    #   V_solvent), so <S^2> belongs to exactly the wavefunction of this SCRF iteration.
    # Caveats: variational wavefunction only (PT does not enter <S^2>); Dice prints RDM
    # elements with |x| > 1e-6 only (error ~1e-6..1e-5); Dice works in a fixed-Ms determinant
    # space without spin adaptation, so <S^2> > S(S+1) signals contamination, which depends
    # on epsilon1 (sweep_epsilon).
    log.info("[PASS 2] <S^2>: Dice fullrestart from the pass-1 variational space, PT switched off")
    log.info("[PASS 2] NOTE: this pass only regenerates the variational RDM; its energy is NOT the reported SHCI energy.")
    mch.stochastic = True
    mch.nPTiter = 0
    # separate input/output files, so the pass-1 Dice output stays untouched
    mch.configFile = (res_str+f"SHCI_input_{shift_fname_core}_S2.dat")
    mch.outputFile = (res_str+f"SHCI_output_{shift_fname_core}_S2.dat")
    start_time_S2 = time.time()
    SHCI_var_res = mch.kernel(res['H1'],
                              res['H2'],
                              norb=res['NORB'],
                              nelec=nelec,
                              # in pyscf.shciscf the restart is controlled by this argument (not
                              # by mch.fullrestart): True writes "fullrestart" into the Dice input
                              fciRestart=True,
                              ecore=res['ECORE'])
    SHCI_S2_runtime = time.time() - start_time_S2
    SHCI_E_var = SHCI_var_res[0]  # nPTiter=0 -> variational active-space energy (ECORE = 0)

    dm1_cas, dm2_cas = mch.make_rdm12(0, res['NORB'], nelec)
    n_act_elec = nelec[0] + nelec[1]
    SHCI_S2 = -n_act_elec * (n_act_elec - 4) / 4 - 0.5 * np.einsum('pqqp->', dm2_cas)
    SHCI_S2_exact = (spin_number / 2) * (spin_number / 2 + 1)

    # Sanity check: the variational energy re-evaluated from the RDMs must reproduce Dice's
    # variational energy; otherwise the RDM on disk is not the variational one and <S^2> is
    # invalid. Both are active-space energies with the same h1e (incl. V_solvent) and ECORE.
    SHCI_E_var_rdm = (res['ECORE'] + np.einsum('pq,pq->', res['H1'], dm1_cas)
                      + 0.5 * np.einsum('pqrs,pqrs->', res['H2'], dm2_cas))
    E_RDM_TOL = 1e-4  # Ha; loose because Dice truncates printed RDM elements at 1e-6
    rdm_check_ok = abs(SHCI_E_var_rdm - SHCI_E_var) < E_RDM_TOL
    if not rdm_check_ok:
        log.warning(f"[PASS 2] WARNING: E_var from RDMs ({SHCI_E_var_rdm:.10f}) differs from Dice "
                    f"E_var ({SHCI_E_var:.10f}) by {SHCI_E_var_rdm - SHCI_E_var:.3e} Ha > "
                    f"{E_RDM_TOL:.0e} Ha. The RDM is NOT the variational one -> <S^2> is NOT reliable!")

    # Summary lines (also written to the run log by the SCRF loop; C1.ipynb parses "S^2 = ").
    # Energies here are ACTIVE-SPACE energies (ECORE = 0), not total energies.
    s2_summary = (
        f"\nS^2 = {SHCI_S2:.8f}\n"
        f"S(S+1) exact = {SHCI_S2_exact:.8f}\n"
        f"S^2 deviation = {SHCI_S2 - SHCI_S2_exact:.3e}\n"
        f"E_var active (Dice, pass 2) = {SHCI_E_var:.10f}\n"
        f"E_var active from RDMs = {SHCI_E_var_rdm:.10f}\n"
        f"RDM energy check passed = {rdm_check_ok}\n"
        f"# <S^2> of the variational SHCI wavefunction (no PT contribution), from the spin-free 2-RDM\n"
        f"# of a PT-free fullrestart (see SHCI_output_{shift_fname_core}_S2.dat).\n"
    )
    # append to the pass-1 output so energy and <S^2> of the (last) SCRF iteration live in one file
    with open(os.path.join(mch.runtimeDir, main_output_file), 'a') as f:
        f.write(s2_summary)

    # =========================================================================
    # CLEANUP of this SCRF iteration (pass 2 and all RDM reads are done)
    # =========================================================================
    # Every SCRF iteration starts with a fresh (non-restart) pass 1, so nothing below is
    # needed by the next iteration:
    #   removed  SHCI_FCIDUMP_{shift}      active-space integrals of this iteration (rewritten
    #                                      by the next one; reproducible from ROHF + AVAS)
    #   removed  {root}-hamiltonian.bkp    sparse H of the variational space (largest file)
    #   removed  {root}-helpers.bkp        determinant lookup tables used to build H
    #   removed  {root}-spatialRDM.bkp     binary (Dice-only) duplicate of spatialRDM.*.txt
    #   removed  {root}-variational.bkp    determinants + CI coefficients, only needed for a
    #                                      Dice restart (pass 2); KEEP_VARIATIONAL_BKP = True keeps it
    #   kept     spatialRDM / spatial1RDM / spin1RDM .{r}.{r}.txt and shci.e of this iteration
    #            (after the last iteration: the variational RDMs of the final SCRF iteration)
    KEEP_VARIATIONAL_BKP = False
    files_to_remove = [Path(mch.runtimeDir) / mch.integralFile]
    scratch_path = Path(mch.scratchDirectory)
    for pattern in ['*-hamiltonian.bkp', '*-helpers.bkp', '*-spatialRDM.bkp']:
        files_to_remove += list(scratch_path.glob(pattern))
    if not KEEP_VARIATIONAL_BKP:
        files_to_remove += list(scratch_path.glob('*-variational.bkp'))
    freed_bytes = 0
    for fpath in files_to_remove:
        if fpath.exists():
            freed_bytes += fpath.stat().st_size
            fpath.unlink()
    log.info("[CLEANUP] removed FCIDUMP + Dice *.bkp files (%.1f MB), kept RDM text files in %s",
             freed_bytes / 1024**2, scratch_path)

    init_qchem_info = {}
    init_qchem_info['SHCI_energy'] = SHCI_res[0]          # variational + PT active-space energy (pass 1)
    init_qchem_info['SHCI_S2'] = SHCI_S2                  # variational <S^2> (pass 2)
    init_qchem_info['SHCI_S2_exact'] = SHCI_S2_exact
    init_qchem_info['SHCI_E_var'] = SHCI_E_var            # variational active-space energy (pass 2)
    init_qchem_info['SHCI_E_var_rdm'] = SHCI_E_var_rdm    # sanity check, should equal SHCI_E_var
    init_qchem_info['SHCI_rdm_check_ok'] = rdm_check_ok
    init_qchem_info['SHCI_metadata'] = SHCI_metadata      # pass-1 (production) solver settings
    init_qchem_info['SHCI_runtime'] = SHCI_runtime        # pass 1 only
    init_qchem_info['SHCI_S2_runtime'] = SHCI_S2_runtime  # pass 2 only
    init_qchem_info['S2_summary'] = s2_summary

    return init_qchem_info, SHCI_res, rdm1_cas


# %%
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


def print_summary(
    scrf_data: list[dict],
    reference_energy: float = None,
    *,
    num_elec_a: int,
    num_elec_b: int,
    num_orbitals: int,
    logger,
):
    """Print a concise summary table of the SCRF iterations.

    The active-space dimensions and SQD batching parameters are passed in
    explicitly (rather than read from globals as in the original notebook)
    so this function is fully reusable.
    """
    logger.info("\n" + "=" * 80)
    logger.info("  SQD IEF-PCM POST-PROCESSING SUMMARY")
    logger.info("=" * 80)
    logger.info(f"  Active space       : ({num_elec_a + num_elec_b}e, {num_orbitals}o)")
    logger.info(f"  SCRF iterations    : {len(scrf_data)}")

    total_wall = sum(d["wall_time"] for d in scrf_data)
    logger.info(f"  Total wall time    : {total_wall / 60:.1f} min")

    logger.info("-" * 80)
    header = (
        f"  {'SCRF':>4s}  {'E_active (Ha)':>18s}  {'ecore (Ha)':>16s}"
        f"  {'E_total (Ha)':>18s}  {'E_solvent (Ha)':>16s}"
    )
    if reference_energy is not None:
        header += f"  {'Error (mHa)':>12s}"
    logger.info(header)
    logger.info("-" * 80)

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
        logger.info(line)

    logger.info("-" * 80)
    final_e = scrf_data[-1]["e_total"]
    logger.info(f"  Final E_total      : {final_e:.10f} Ha")

    if reference_energy is not None:
        final_err = abs(final_e - reference_energy)
        logger.info(f"  Reference energy   : {reference_energy:.10f} Ha")
        logger.info(f"  Final error        : {final_err:.6e} Ha"
              f"  ({final_err * 627.509:.4f} kcal/mol)")
        if final_err < 1.6e-3:
            logger.info("  [OK] Within chemical accuracy (1 kcal/mol ~ 1.6 mHa)")
        else:
            logger.info("  [!!] Not at chemical accuracy — consider more SCRF iters"
                  " or larger samples_per_batch")

    # Solvation free energy estimate
    if len(scrf_data) >= 2:
        # Compare final solvated energy to first (gas-phase-like) energy
        delta_g = scrf_data[-1]["e_total"] - scrf_data[0]["e_total"]
        logger.info(f"\n  Approximate solvation correction: {delta_g:.6f} Ha"
              f"  ({delta_g * 627.509:.4f} kcal/mol)")
        logger.info("  (NOTE: for a rigorous G_solv, compare to a separate gas-phase"
              " SQD calculation.)")

    logger.info("=" * 80 + "\n")

# %%
if Path(os.getcwd()).name != "sqd_evaluation":
    os.chdir("../../../")
os.getcwd()

# %%
import sys

import time
import pyscf.gto
from pyscf import data, lib
from pyscf.shciscf import shci
import pickle
from pyscf import tools

print("CPU counts: ", os.cpu_count(), "| threads used by this run (PySCF + Dice): ", n_threads)


shift_array = ['-0.15', '0.0', '0.15', '0.3', '0.45', '0.6', '0.75', '0.9', '1.05', '1.2']
shift = sys.argv[1]

spin_number = int(sys.argv[2])
n_so = int(sys.argv[3]) # number of spin-orbitals # 16, 24 | 17, 25, (23)
# Optional 4th argument; the default keeps the old 3-argument calls (charge +2) working.
charge = int(sys.argv[4]) if len(sys.argv) > 4 else 2
medium = 'solvent'
basis_set = "def2-TZVP"

# Variational threshold epsilon1 (Dice sweep_epsilon, used in run_SHCI); SHCI_EPS1 is exported by
# run_SHCI_concurrent.sh. The default 1e-3 is the published setting (exp_301); any other value is a new
# experiment exp_302_..._eps1_<value>.
eps1_str = os.environ.get("SHCI_EPS1", "1e-3")
eps1 = float(eps1_str)

# '_S2' suffix: runs that also compute <S^2> (see module docstring, "History").
# Kept separate so the energy-only results ('..._det', '..._det_v2') are not overwritten.
if eps1 == 1e-3:
    exp_name = f'paper_v3_exp_301_{medium}_det_S2'
else:
    exp_name = f'paper_v3_exp_302_{medium}_det_S2_eps1_{eps1_str}'

num_orbitals = n_so // 2

# Number of active electrons, keyed by charge -> n_so -> spin_number (2S).
# Charge +2: Co(II) Doublet/Quartet; charge +3: Co(III) Singlet/Quintet.
# Same values as in the gas script and the results notebooks (1.results_charge_2/3.ipynb).
n_elec = { # first charge, then n_qubits, then spin value
    2: {
        34: {1: 23, 3: 23},
    },
    3: {
        34: {0: 22, 4: 22},
        38: {0: 24, 4: 24},
        46: {0: 28, 4: 28},
        50: {0: 30, 4: 30},
    },
}
num_elec_a, num_elec_b = ((n_elec[charge][n_so][spin_number] + spin_number)//2,
                          (n_elec[charge][n_so][spin_number] - spin_number)//2)
num_qubits = num_orbitals * 2
n_scrf_iter: int          = 4     # outer SCRF (reaction-field) iterations
as_label = 'norb_%d_nelec_%d_%d' % (num_orbitals, num_elec_a, num_elec_b)
print("Active space label: ", as_label)

charge_sign = '_plus' if charge>0 else "_minus" if charge<0 else ""
spin_states = {
    0: "Singlet",
    1: "Doublet",
    3: "Quartet",
    4: "Quintet",
}
spin_state = spin_states[spin_number]
shift_sign = 'minus_' if shift[0]=='-' else ''
shift_fname_core = shift_sign + shift.strip('-').split('.')[0] + '_' + shift.split('.')[1][::-1].zfill(2)[::-1]

# %%
CLASSICAL_DATA_DIR = (f'./data/interim/metalloenzymes/Co_5_H2O_CO2/MF/ROHF_stab_analysis_5/{medium}/charge{charge}_spin{spin_number}')

DATA_DIR = Path('./data/raw/metalloenzymes/Co_5_H2O_CO2_AVAS/')
mol_name = f'Co_5H2O_CO2_rhf_optimized_{shift}_shifted.xyz'
mol_path = DATA_DIR / mol_name
# fcidump_path = list(Path(f'./data/interim/metalloenzymes/Co_5_H2O_CO2/MF/ROHF_stab_analysis_5/{medium}/charge{charge}_spin{spin_number}/active_spaces/shift_{shift}/').glob(f'spin_{spin_number}_shift_{shift}_norb_{n_so//2}*.fcidump'))[0]

# Load the FCIDUMP file
# res = tools.fcidump.read(fcidump_path)
mol = pyscf.gto.Mole(
    atom=mol_path.__str__(),
    symmetry=False,
    spin=spin_number,
    charge=charge,
    verbose=0,
    basis=basis_set
)
# mol.output = RESULTS_DIR.joinpath(f'ROHF_shift_{shift_fname_core}.out')
mol.build()
print(mol.spin, mol.charge, mol.nelectron, mol.nao_nr())


RESULTS_DIR = Path(f'./data/interim/metalloenzymes/Co_5_H2O_CO2/SHCI/{basis_set}/AVAS/{medium}/{exp_name}/{spin_state}{charge_sign}_{charge}/{n_so}qubits/')
Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
# Provenance: copy this script into the results dir on EVERY run, so the copy always matches
# the version that produced the latest results (copying only when the dir is first created
# left stale copies after later edits). Written to a per-process temp file and then renamed
# with os.replace (atomic), so concurrent jobs of the same exp_name/spin/n_so never leave a
# half-written copy. Fails softly (e.g. __file__ undefined when run cell-by-cell).
try:
    current_file = Path(__file__)
    destination_file = RESULTS_DIR / current_file.name
    tmp_file = RESULTS_DIR / f".{current_file.name}.{os.getpid()}.tmp"
    tmp_file.write_text(current_file.read_text())
    os.replace(tmp_file, destination_file)
except Exception as e:
    print(f"Failed to copy the script into {RESULTS_DIR} for provenance: {e}")

SCRATCH_DIR = Path(f'./notebooks/analysis/SHCI/scratch/{medium}/{exp_name}/{spin_state}{charge_sign}_{charge}/{n_so}qubits/{shift_fname_core}/')
SCRATCH_DIR.mkdir(exist_ok=True, parents=True)

# %%
# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
fh = logging.FileHandler(RESULTS_DIR.joinpath(f'HCI_shift_{shift_fname_core}.log'),)
log.addHandler(fh)
log.info(f"Active space label: {as_label}")

# 2.  Load classical (pre-quantum) inputs
mf, mc, h1e_cas, h2e_cas, ecore = load_converged_scf_and_integrals(
    CLASSICAL_DATA_DIR,
    shift,
    medium,
    as_label=as_label,
    num_orbitals=num_orbitals,
    num_elec_a=num_elec_a,
    num_elec_b=num_elec_b,
)

# Classical reference (CCSD total energy) at this geometry shift
reference_energy = load_reference_energy(
    CLASSICAL_DATA_DIR, as_label, shift_array.index(shift)
)

if not hasattr(mc, "with_solvent"):
    raise AttributeError(
        "The CASCI object does not have a solvent model.\n"
        "Create it with e.g.:  mc = solvent.ddCOSMO(mc)  or  mc.PCM()"
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

    res = {
        'NELEC': n_elec[charge][n_so][spin_number],
        'NORB': num_orbitals,
        'H1': h1e_cas,
        'H2': h2e_cas,
        'ECORE': 0,  # not used in this context but required by the SHCI interface
    }
    log.info(res['NELEC'])
    
    t_start = time.time()
    # ── Step 3: inner S-CORE + batched diagonalisation ────────────────
    log.info("Starting HCI post-processing (%d orbitals, %d↑ + %d↓ electrons)",
                num_orbitals, num_elec_a, num_elec_b)
    
    
    res_str = f'../../../../data/interim/metalloenzymes/Co_5_H2O_CO2/SHCI/{basis_set}/AVAS/{medium}/{exp_name}/{spin_state}{charge_sign}_{charge}/{n_so}qubits/'
    scratch_dir = f"/{medium}/{exp_name}/{spin_state}{charge_sign}_{charge}/{n_so}qubits/{shift_fname_core}"
    
    init_qchem_info, SHCI_res, rdm1_cas = run_SHCI(mol, res, shift_fname_core, res_str, scratch_dir, spin_number)
    rdm1_cas = rdm1_cas[0]+rdm1_cas[1]
    
    import pickle as pkl
    pkl.dump(
        init_qchem_info,
        open(RESULTS_DIR.joinpath(f'shift_{shift_fname_core}_results_k_{k}.pkl'), 'wb')
    )
    pkl.dump(
        SHCI_res,
        open(RESULTS_DIR.joinpath(f'shift_{shift_fname_core}_results_rawk_{k}.pkl'), 'wb')
    )
    
    wt = time.time() - t_start
    log.info("HCI complete in %.1f min  │  Best E_total = %.10f Ha",
            wt / 60.0, SHCI_res[0] + ecore)
    # <S^2> of this SCRF iteration (pass 2 in run_SHCI). The "S^2 = " line is what C1.ipynb
    # parses from this log; the LAST one in the file = final SCRF iteration of the latest run.
    log.info("  <S^2> of SCRF iteration %d (variational SHCI wavefunction, no PT):", k)
    for s2_line in init_qchem_info['S2_summary'].strip().splitlines():
        log.info("  " + s2_line if s2_line.startswith('#') else s2_line)

    total_wall += wt
    e_active = SHCI_res[0]
    
    
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
        "wall_time":      wt,
        # <S^2> of this iteration's variational wavefunction (pass 2) + its sanity check
        "s2":               float(init_qchem_info['SHCI_S2']),
        "s2_exact":         float(init_qchem_info['SHCI_S2_exact']),
        "e_var_active":     float(init_qchem_info['SHCI_E_var']),      # active space, ECORE = 0
        "e_var_active_rdm": float(init_qchem_info['SHCI_E_var_rdm']),
        "rdm_check_ok":     bool(init_qchem_info['SHCI_rdm_check_ok']),
        "s2_wall_time":     float(init_qchem_info['SHCI_S2_runtime']),
    })

log.info("=" * 70)
log.info(
    "SCRF loop complete in %.1f min  |  Final E_total = %.10f Ha",
    total_wall / 60.0, scrf_data[-1]["e_total"],
)
log.info(
    "Final S^2 = %.8f  (S(S+1) = %.4f, deviation %.3e, RDM check passed: %s) -- final SCRF iteration",
    scrf_data[-1]["s2"], scrf_data[-1]["s2_exact"],
    scrf_data[-1]["s2"] - scrf_data[-1]["s2_exact"], scrf_data[-1]["rdm_check_ok"],
)
log.info("  S^2 per SCRF iteration: %s", ", ".join(f"{d['s2']:.6f}" for d in scrf_data))
log.info("=" * 70)


# ── Save results ──────────────────────────────────────────────────────────
# Save the SCRF data (minus large arrays for portability)
out_pkl = RESULTS_DIR.joinpath(f'HCI_shift_{shift_fname_core}_results_raw.pkl')
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
out_json = RESULTS_DIR.joinpath(f'HCI_shift_{shift_fname_core}_results_summary.json')
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
        "s2":             d["s2"],
        "s2_exact":       d["s2_exact"],
        "rdm_check_ok":   d["rdm_check_ok"],
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
    logger=log
)

# ── Plot convergence ──────────────────────────────────────────────────────
scrf_plot_path = RESULTS_DIR.joinpath(f'HCI_shift_{shift_fname_core}_convergence.png')
plot_scrf_convergence(scrf_data, reference_energy, save_path=scrf_plot_path)
log.info("Done.")

