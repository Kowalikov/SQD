"""
SHCI (Dice) reference calculation for the Co(5H2O)CO2 complex in the gas phase,
including the spin purity <S^2> of the SHCI wavefunction.

Usage (run from notebooks/analysis/SHCI/, see run_SHCI.sh):
    python SHCI_gas_v3_new_handlers.py <shift> <spin_number> <n_so> [<charge>]

    shift        CO2 displacement as in the .xyz file names, e.g. '-0.15', '0.0', '1.2'
    spin_number  2S = n_alpha - n_beta: 0 Singlet, 1 Doublet, 3 Quartet, 4 Quintet
    n_so         number of active spin-orbitals (= qubits), n_orb = n_so // 2
    charge       total charge, optional, default 2 (charge 3 -> Singlet/Quintet)

What the script does
--------------------
1. Loads the converged ROHF (density-fitted) and AVAS active-space orbitals from
   data/interim/.../MF/ROHF_stab_analysis_5/{medium}/charge{charge}_spin{spin}/
   and builds the active-space integrals (h1e, h2e, ecore) via CASCI.
2. PASS 1 (production run): variational HCI + deterministic PT (epsilon2) with
   Dice. The reported SHCI energy (`SHCI_energy`) is the variational + PT energy
   of this pass. Nothing about this pass was changed when <S^2> was added.
3. PASS 2 (<S^2> run): restarts Dice from the converged variational determinant
   space of pass 1 with PT switched off, only to obtain the *variational* spin-free
   2-RDM, and evaluates <S^2> from it (details and reasons in the <S^2> section
   below).
4. Appends the <S^2> summary to the pass-1 Dice output (SHCI_output_{shift}.dat),
   and stores everything in shift_{shift}_results.pkl.
5. Cleanup: deletes the FCIDUMP and the large Dice scratch backups (*.bkp);
   keeps the variational RDM text files + shci.e in the scratch dir for
   diagnostics (see the CLEANUP section at the end).

Outputs (in data/interim/.../SHCI/{basis}/AVAS/gas/{exp_name}/{Spin}_plus_{charge}/{n_so}qubits/)
    SHCI_FCIDUMP_{shift}           active-space integrals passed to Dice (DELETED at the end)
    SHCI_input_{shift}.dat        Dice input,  pass 1 (production, with PT)
    SHCI_output_{shift}.dat        Dice output, pass 1 + appended <S^2> summary lines:
                                       "S^2 = ...", "S(S+1) exact = ...",
                                       "S^2 deviation = ...", "E_var from RDMs = ..."
                                   (parsed by notebooks/.../paper/JCTC_review/C1.ipynb)
    SHCI_input_{shift}_S2.dat      Dice input,  pass 2 (fullrestart, no PT)
    SHCI_output_{shift}_S2.dat     Dice output, pass 2
    shift_{shift}_results.pkl      dict with keys:
        SHCI_energy      variational + PT energy of pass 1 [Ha]  <- energy used in the paper
        SHCI_S2          <S^2> of the variational SHCI wavefunction (no PT contribution)
        SHCI_S2_exact    S(S+1) of the target spin state
        SHCI_E_var       variational energy returned by Dice in pass 2 [Ha]
        SHCI_E_var_rdm   variational energy re-evaluated from the pass-2 RDMs [Ha] (sanity check)
        SHCI_rdm1        spin-summed variational 1-RDM in the active space (n_orb x n_orb)
        SHCI_metadata    vars() of the SHCI solver for pass 1 (production settings)
        SHCI_runtime     wall time of pass 1 [s]
        SHCI_S2_runtime  wall time of pass 2 [s]
    shift_{shift}_results_raw.pkl  raw return value of mch.kernel() of pass 1

History
-------
- exp_name 'paper_v2_exp_301_gas_new_loader': energies only (no <S^2>).
- exp_name 'paper_v2_exp_301_gas_new_loader_S2': same Dice settings, same PT
  energies (checked bit-identical for Doublet +2, 34 qubits, shift 0.0), plus
  <S^2>; added to answer the JCTC review comment on spin purity (SI S1.7).
"""
import os, sys

# ----- threads -----
# Must be set BEFORE numpy/pyscf are imported: BLAS/OpenMP read these variables
# only once, at library load. The Dice subprocess inherits them (Dice is built
# with OpenMP), so they also set the Dice thread count.
# Values already exported by the caller win (run_SHCI_concurrent.sh exports
# OMP_NUM_THREADS=<threads per job> so concurrent jobs don't oversubscribe the
# CPU); otherwise all cores are used, as for the serial run_SHCI.sh.
n_threads = int(os.environ.get("OMP_NUM_THREADS", os.cpu_count()))
for var in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ.setdefault(var, str(n_threads))

import numpy as np
from pathlib import Path

import time
import pyscf.gto
from pyscf import data, lib
from pyscf.shciscf import shci
import pickle
from pyscf import tools

os.chdir('../../../')
print(os.getcwd())


print("CPU counts: ", os.cpu_count(), "| threads used by this run (PySCF + Dice): ", n_threads)


shift = sys.argv[1]
spin_number = int(sys.argv[2])
n_so = int(sys.argv[3]) # number of spin-orbitals # 16, 24 | 17, 25, (23)
# Optional 4th argument; the default keeps the old 3-argument calls (charge +2) working.
charge = int(sys.argv[4]) if len(sys.argv) > 4 else 2
medium = 'gas'
exp_name = 'paper_v2'
# '_S2' suffix: runs that also compute <S^2> (see module docstring, "History").
# Kept separate so the original 'paper_v2_exp_301_gas_new_loader' results are not overwritten.
exp_name += f'_exp_301_{medium}_new_loader_S2'

basis_set = "def2-TZVP"
CLASSICAL_DATA_DIR = (f'./data/interim/metalloenzymes/Co_5_H2O_CO2/MF/ROHF_stab_analysis_5/{medium}/charge{charge}_spin{spin_number}')


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

# Number of active electrons, keyed by charge -> n_so -> spin_number (2S).
# Charge +2: Co(II) Doublet/Quartet; charge +3: Co(III) Singlet/Quintet.
# Same values as in the results notebooks (1.results_charge_2/3.ipynb).
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
print("Num electrons (alpha, beta): ", num_elec_a, num_elec_b)
as_label = 'norb_%d_nelec_%d_%d' % (n_so//2, num_elec_a, num_elec_b)
print("Active space label: ", as_label)


DATA_DIR = Path('./data/raw/metalloenzymes/Co_5_H2O_CO2_AVAS/')
mol_name = f'Co_5H2O_CO2_rhf_optimized_{shift}_shifted.xyz'
mol_path = DATA_DIR / mol_name


RESULTS_DIR = Path(f'./data/interim/metalloenzymes/Co_5_H2O_CO2/SHCI/{basis_set}/AVAS/{medium}/{exp_name}/{spin_state}{charge_sign}_{charge}/{n_so}qubits/')
if not os.path.exists(RESULTS_DIR):
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    # copy current file into this directory for provenance
    current_file = Path(__file__)
    destination_file = RESULTS_DIR / current_file.name
    try:
        with open(current_file, 'r') as src, open(destination_file, 'w') as dst:
            dst.write(src.read())
        
    except Exception as e:
        print(f"Failed to copy script {current_file} to {destination_file}: {e}")

SCRATCH_DIR = Path(f'./notebooks/analysis/SHCI/scratch/{medium}/{exp_name}/{spin_state}{charge_sign}_{charge}/{n_so}qubits/{shift_fname_core}/')
SCRATCH_DIR.mkdir(exist_ok=True, parents=True)
# Runtime dir: only the runtime/ base (mch.runtimeDir, from pyscf.shciscf settings) is used.
# Dice file names are given relative to it (res_str = '../../../../data/...'), so all Dice
# files land in RESULTS_DIR and the base itself stays empty; it only has to exist for the
# '../' path to resolve. No per-run runtime sub-directories are created: earlier versions
# created runtime/{medium}/{exp_name}/.../{shift}/ (always empty) and removed it at the end,
# which races between concurrent jobs (one job's cleanup removing a parent dir while another
# job creates its own) -> dropped for run_SHCI_concurrent.sh.
Path('./notebooks/analysis/SHCI/runtime').mkdir(exist_ok=True, parents=True)


mol = pyscf.gto.Mole(
    atom=mol_path.__str__(),
    symmetry=False,
    spin=spin_number,
    charge=charge,
    verbose=0,
    basis=basis_set
)
mol.build()


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
    from pyscf.scf import chkfile
    from pyscf import mcscf, ao2mo, lib

    
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


mch = shci.SHCI(mol)
# To choose parameters, look into: 
# - https://sanshar.github.io/Dice/keywords.html
# - https://github.com/pyscf/shciscf/blob/master/pyscf/shciscf/shci.py
# parallel optimization
mch.mpiprefix = '' #'mpirun -np 5'  

# variational HCISCI run
mch.dE = 1.0e-8 # The energy tolerance used to control convergence during the variational HCI algorithm.
mch.davidsonTol = 5.0e-5
mch.maxIter = 20

# mch.epsilon1 = 1e-3
mch.sweep_iter = [20] # [5, 20] - from 0-4th first sweep epsilon, then up to 19th second
mch.sweep_epsilon = [ 1e-3]  #TODO [ 1e-4] #[ 1e-4, 3e-6]

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

# DoRDM: Dice writes the spin-free (spatial) 2-RDM to
# {scratchDirectory}/spatialRDM.0.0.txt (read back with mch.make_rdm12).
# CAUTION: with deterministic PT (stochastic=False) Dice OVERWRITES this file
# at the end of the PT step with a PT-response RDM, i.e. after pass 1 the file
# does NOT hold the variational RDM. This is why <S^2> needs pass 2 below.
mch.DoRDM = True


res_str = f'../../../../data/interim/metalloenzymes/Co_5_H2O_CO2/SHCI/{basis_set}/AVAS/{medium}/{exp_name}/{spin_state}{charge_sign}_{charge}/{n_so}qubits/'
mch.integralFile = (res_str+f"SHCI_FCIDUMP_{shift_fname_core}")
mch.configFile =  (res_str+f"SHCI_input_{shift_fname_core}.dat")
mch.outputFile = (res_str+f"SHCI_output_{shift_fname_core}.dat")
mch.scratchDirectory += f"/{medium}/{exp_name}/{spin_state}{charge_sign}_{charge}/{n_so}qubits/{shift_fname_core}"
mch.printbestdeterminants = True


# 2.  Load classical (pre-quantum) inputs
mf, mc, h1e_cas, h2e_cas, ecore = load_converged_scf_and_integrals(
    CLASSICAL_DATA_DIR,
    shift,
    medium,
    as_label=as_label,
    num_orbitals=n_so//2,
    num_elec_a=num_elec_a,
    num_elec_b=num_elec_b,
)

# =============================================================================
# PASS 1: production SHCI run (variational HCI + deterministic PT)
# -> SHCI_res[0] is the energy reported in the paper (SHCI_energy).
# =============================================================================
print("\n[PASS 1] Production SHCI run: variational HCI + deterministic PT")
start_tmie = time.time()
SHCI_res = mch.kernel(h1e_cas,
                      h2e_cas,
                      norb=n_so//2,
                      nelec=(num_elec_a, num_elec_b),
                      fciRestart=False,
                      ecore=ecore)

end_time = time.time()
SHCI_runtime = end_time - start_tmie
print(f"[PASS 1] SHCI energy (variational + PT): {SHCI_res[0]:.10f} Ha  ({SHCI_runtime:.1f} s)")

# Snapshot the solver settings NOW: pass 2 changes stochastic/nPTiter/file names,
# and the stored metadata must describe the production run (pass 1).
mch.stdout = mol.output
SHCI_metadata = dict(vars(mch))  # snapshot of the production (PT) run settings
main_output_file = mch.outputFile

# =============================================================================
# PASS 2: <S^2> of the variational SHCI wavefunction
# =============================================================================
# Why not simply mch.spin_square()?
#   pyscf.shciscf.shci.SHCI.spin_square() does NOT compute anything: it returns
#   the nominal S(S+1) from (n_alpha - n_beta)/2. It cannot detect spin
#   contamination, so it must not be used to report spin purity.
#
# How <S^2> is computed here:
#   From the spin-free 2-RDM in PySCF chemist convention,
#       dm2[p,q,r,s] = sum_{sigma,tau} < a^+_{p sigma} a^+_{r tau} a_{s tau} a_{q sigma} >,
#   one has (N = number of active electrons)
#       <S^2> = -N(N-4)/4 - 1/2 * sum_pq dm2[p,q,q,p].
#   Only active-space electrons contribute: the core is closed-shell (S=0).
#   The formula was validated against pyscf.fci.spin_op.spin_square on O2/6-31G
#   CASCI(6o) roots for 2S = 0, 1, 2, 4 (incl. spin-contaminated roots, e.g.
#   <S^2> = 2.0 or 3.75 in the Ms = 0 / 1/2 sectors); agreement to 1e-8.
#
# Why a second Dice run:
#   With deterministic PT + DoRDM, Dice overwrites spatialRDM.0.0.txt at the end
#   of PT with a PT-response RDM. That is NOT an expectation-value RDM; using it
#   gave <S^2> = 0.880 and an energy 0.14 Ha below the variational one (Doublet
#   +2, 34 qubits, shift 0.0), i.e. meaningless. Dice has no keyword to skip
#   deterministic PT, so pass 2 restarts from the converged determinant space
#   of pass 1 ("fullrestart", reads {scratch}/0-variational.bkp) with
#       stochastic = True, nPTiter = 0
#   which (a) switches PT off (documented PySCF/Dice way) and (b) makes Dice
#   skip the PT-RDM step ("We cannot perform PT RDM with stochastic PT.
#   Disabling RDM." in the pass-2 output), so the variational RDM written after
#   the Davidson step stays on disk. The pass-2 cost is ~2 Davidson iterations.
#
# Caveats (state them in the SI when reporting these numbers):
#   - <S^2> refers to the VARIATIONAL SHCI wavefunction; the PT correction is
#     an energy correction only and does not enter <S^2>.
#   - Dice prints RDM elements with |x| > 1e-6 only, so <S^2> and E_var_rdm
#     carry an error of order 1e-6 .. 1e-5 (observed |dE| ~ 4e-7 Ha).
#   - Dice works in a fixed Ms = 2S/2 determinant space without spin
#     adaptation, so <S^2> > S(S+1) signals contamination from higher spin
#     states; its size depends on epsilon1 (sweep_epsilon).
print("\n[PASS 2] <S^2>: Dice fullrestart from the pass-1 variational space, PT switched off")
print("[PASS 2] NOTE: this pass only regenerates the variational RDM; its energy is NOT the reported SHCI energy.")
mch.stochastic = True
mch.nPTiter = 0
# separate input/output files, so the pass-1 Dice output stays untouched
mch.configFile = (res_str+f"SHCI_input_{shift_fname_core}_S2.dat")
mch.outputFile = (res_str+f"SHCI_output_{shift_fname_core}_S2.dat")
start_time_S2 = time.time()
SHCI_var_res = mch.kernel(h1e_cas,
                          h2e_cas,
                          norb=n_so//2,
                          nelec=(num_elec_a, num_elec_b),
                          # In pyscf.shciscf the restart is controlled by this argument
                          # (not by mch.fullrestart): True writes "fullrestart" + "maxiter 10"
                          # into the Dice input, so the determinant space of pass 1 is reused.
                          fciRestart=True,
                          ecore=ecore)
SHCI_S2_runtime = time.time() - start_time_S2
SHCI_E_var = SHCI_var_res[0]  # nPTiter=0 -> no PT correction, i.e. the variational energy

# Read the variational spin-free RDMs (mch.make_rdm12 reads {scratch}/spatialRDM.0.0.txt;
# dm1 is obtained from the dm2 partial trace, normalised to N).
dm1_cas, dm2_cas = mch.make_rdm12(0, n_so//2, (num_elec_a, num_elec_b))
n_act_elec = num_elec_a + num_elec_b
SHCI_S2 = -n_act_elec * (n_act_elec - 4) / 4 - 0.5 * np.einsum('pqqp->', dm2_cas)
SHCI_S2_exact = (spin_number / 2) * (spin_number / 2 + 1)

# Sanity check: the variational energy re-evaluated from the RDMs must reproduce
# Dice's variational energy. If not, the RDM on disk is not the variational one
# (e.g. a PT-response RDM, or a stale file from another run), and <S^2> is invalid.
SHCI_E_var_rdm = ecore + np.einsum('pq,pq->', h1e_cas, dm1_cas) + 0.5 * np.einsum('pqrs,pqrs->', h2e_cas, dm2_cas)
E_RDM_TOL = 1e-4  # Ha; loose because Dice truncates printed RDM elements at 1e-6
rdm_check_ok = abs(SHCI_E_var_rdm - SHCI_E_var) < E_RDM_TOL
if not rdm_check_ok:
    print(f"[PASS 2] WARNING: E_var from RDMs ({SHCI_E_var_rdm:.10f}) differs from Dice E_var "
          f"({SHCI_E_var:.10f}) by {SHCI_E_var_rdm - SHCI_E_var:.3e} Ha > {E_RDM_TOL:.0e} Ha. "
          f"The RDM is NOT the variational one -> <S^2> below is NOT reliable!")

# Summary lines; the "S^2 = <value>" line is what C1.ipynb parses (last match in the file).
s2_summary = (
    f"\nS^2 = {SHCI_S2:.8f}\n"
    f"S(S+1) exact = {SHCI_S2_exact:.8f}\n"
    f"S^2 deviation = {SHCI_S2 - SHCI_S2_exact:.3e}\n"
    f"E_var (Dice, pass 2) = {SHCI_E_var:.10f}\n"
    f"E_var from RDMs = {SHCI_E_var_rdm:.10f}\n"
    f"RDM energy check passed = {rdm_check_ok}\n"
    f"# <S^2> of the variational SHCI wavefunction (no PT contribution), from the spin-free 2-RDM\n"
    f"# of a PT-free fullrestart (see SHCI_output_{shift_fname_core}_S2.dat). The energy reported\n"
    f"# for this run is the PTEnergy above, not E_var.\n"
)
print(s2_summary)
# append to the pass-1 output so energy and <S^2> of a geometry live in one file
with open(os.path.join(mch.runtimeDir, main_output_file), 'a') as f:
    f.write(s2_summary)

init_qchem_info = {}
init_qchem_info['SHCI_energy'] = SHCI_res[0]          # variational + PT (pass 1), reported energy
init_qchem_info['SHCI_S2'] = SHCI_S2                  # variational <S^2> (pass 2)
init_qchem_info['SHCI_S2_exact'] = SHCI_S2_exact
init_qchem_info['SHCI_E_var'] = SHCI_E_var            # variational energy (pass 2)
init_qchem_info['SHCI_E_var_rdm'] = SHCI_E_var_rdm    # sanity check, should equal SHCI_E_var
init_qchem_info['SHCI_rdm_check_ok'] = rdm_check_ok
init_qchem_info['SHCI_rdm1'] = dm1_cas                # spin-summed variational 1-RDM (active space)
init_qchem_info['SHCI_metadata'] = SHCI_metadata      # pass-1 (production) solver settings
init_qchem_info['SHCI_runtime'] = SHCI_runtime        # pass 1 only
init_qchem_info['SHCI_S2_runtime'] = SHCI_S2_runtime  # pass 2 only

# ----- results ----- 
import pickle as pkl
pkl.dump(
    init_qchem_info,
    open(RESULTS_DIR.joinpath(f'shift_{shift_fname_core}_results.pkl'), 'wb')
)
pkl.dump(
    SHCI_res,
    open(RESULTS_DIR.joinpath(f'shift_{shift_fname_core}_results_raw.pkl'), 'wb')
)

print(f"Results saved to {RESULTS_DIR.joinpath(f'shift_{shift_fname_core}_results.pkl')}")

# =============================================================================
# CLEANUP: remove large / reproducible files (only after both passes and the
# pickles are written, since pass 2 needs 0-variational.bkp and the RDMs)
# =============================================================================
# Results dir:
#   SHCI_FCIDUMP_{shift}       active-space integrals; reproducible from the ROHF
#                              checkpoint + AVAS orbitals, so not kept.
# Scratch dir ({mch.scratchDirectory}):
#   removed  {root}-hamiltonian.bkp  sparse H of the variational space (largest file),
#                                    rebuilt by Dice on every run.
#   removed  {root}-helpers.bkp      determinant lookup tables used to build H, rebuilt by Dice.
#   removed  {root}-spatialRDM.bkp   binary (Dice-only) duplicate of spatialRDM.{r}.{r}.txt.
#   removed  {root}-variational.bkp  variational determinants + CI coefficients. Only needed
#                                    for a Dice (full)restart, e.g. pass 2. Set
#                                    KEEP_VARIATIONAL_BKP = True to keep it for later restarts.
#   kept     spatialRDM.{r}.{r}.txt  variational spin-free 2-RDM (after pass 2)  -> <S^2>, E_var
#            spatial1RDM.{r}.{r}.txt variational spin-free 1-RDM                 -> natural orbitals/occupations
#            spin1RDM.{r}.{r}.txt    variational spin-resolved 1-RDM             -> spin densities
#            shci.e                  energy written by Dice (pass 2: variational)
#   (read them back with mch.make_rdm12 / mch.make_rdm1s, pointing scratchDirectory here)
# The Dice input/output .dat files in the results dir are small and always kept.
KEEP_VARIATIONAL_BKP = False

files_to_remove = [Path(mch.runtimeDir) / mch.integralFile]
scratch_dir = Path(mch.scratchDirectory)
for pattern in ['*-hamiltonian.bkp', '*-helpers.bkp', '*-spatialRDM.bkp']:
    files_to_remove += list(scratch_dir.glob(pattern))
if not KEEP_VARIATIONAL_BKP:
    files_to_remove += list(scratch_dir.glob('*-variational.bkp'))

print("\n[CLEANUP] Removing large / reproducible files:")
freed_bytes = 0
for fpath in files_to_remove:
    if fpath.exists():
        freed_bytes += fpath.stat().st_size
        fpath.unlink()
        print(f"  removed {fpath.resolve()}")
print(f"[CLEANUP] Freed {freed_bytes / 1024**2:.1f} MB. Kept RDM text files + shci.e in {scratch_dir}")

# Runtime dir: nothing to clean, no per-run runtime sub-directories are created (see the
# comment where SCRATCH_DIR is set). Leftover empty ones from older runs can be removed with
#   find notebooks/analysis/SHCI/runtime -mindepth 1 -type d -empty -delete

print("Done!")
