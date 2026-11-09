"""
Active-space coupled-cluster benchmarks for [Co(H2O)5CO2]^{2+/3+} (gas phase).

For every geometry along the Co-O(CO2) dissociation path, this script
reconstructs the converged density-fitted (RO)HF reference from its PySCF
checkpoint and replaces the orbitals with the AVAS-rotated ones. It then runs
frozen-core UCCSD and UCCSD(T) inside the selected AVAS active space.

Inputs (inside ``<STATE>/``):
    chk_files/shift_<chk_label>.chk                    converged mean-field solution
    orbitals/shift_<shift>/mo_coeff_<as_label>.npy     AVAS-rotated MO coefficients

Outputs (inside ``<STATE>/``):
    orbitals/shift_<shift>/uccsd_amps_<as_label>.npz   UCCSD amplitudes t1a, t1b, t2aa, t2ab, t2bb
                                                       (used to build the LUCJ circuits)
    active_space_results_classical_<as_label>.npy      pickled dict
                                                       {"<as_label>_ROHF" | "_CCSD" | "_CCSD(T)":
                                                        array of total energies (Ha), one per shift}

``<as_label>`` is ``norb_<norb>_nelec_<n_alpha>_<n_beta>``. ``<chk_label>`` encodes
the shift: 0.0 -> 0_00, 0.3 -> 0_30, -0.15 -> minus_0_15.

Usage:
    Set ``STATE`` and ``NORB`` in the configuration section below, then run
        python classical_calculations.py

    Paths are resolved relative to this file, so the script can be run from any
    working directory. Running it overwrites the output files listed above.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pyscf import cc, lib, scf
from pyscf.scf import chkfile

# =============================================================================
# Configuration
# =============================================================================
MEDIUM = "gas"

STATE = "charge3_spin4"   # charge3_spin0 | charge3_spin4 | charge2_spin1 | charge2_spin3
NORB = 17                 # number of active spatial orbitals; see ACTIVE_SPACES

SHIFTS = ["-0.15", "0.0", "0.15", "0.3", "0.45", "0.6", "0.75", "0.9", "1.05", "1.2"]

# Active spaces used in the paper: state -> {norb: (n_alpha, n_beta)}
ACTIVE_SPACES: dict[str, dict[int, tuple[int, int]]] = {
    "charge3_spin0": {17: (11, 11), 19: (12, 12), 23: (14, 14), 25: (15, 15)},   # Co(III) singlet
    "charge3_spin4": {17: (13, 9), 19: (14, 10), 23: (16, 12), 25: (17, 13)},    # Co(III) quintet
    "charge2_spin1": {17: (12, 11)},                                              # Co(II) doublet
    "charge2_spin3": {17: (13, 10)},                                              # Co(II) quartet
}

METHODS = ("ROHF", "CCSD", "CCSD(T)")

log = logging.getLogger(__name__)


# =============================================================================
# Helpers
# =============================================================================
@dataclass(frozen=True)
class ActiveSpace:
    """AVAS active space: number of spatial orbitals and active (alpha, beta) electrons."""

    norb: int
    n_alpha: int
    n_beta: int

    @property
    def n_elec(self) -> int:
        return self.n_alpha + self.n_beta

    @property
    def label(self) -> str:
        return f"norb_{self.norb}_nelec_{self.n_alpha}_{self.n_beta}"


def get_active_space(state: str, norb: int) -> ActiveSpace:
    """Look up the active space for ``state`` with ``norb`` orbitals, or raise if it was not used."""
    if state not in ACTIVE_SPACES:
        raise ValueError(f"Unknown STATE {state!r}; choose one of {sorted(ACTIVE_SPACES)}.")
    if norb not in ACTIVE_SPACES[state]:
        raise ValueError(
            f"NORB={norb} is not available for {state}; choose one of {sorted(ACTIVE_SPACES[state])}."
        )
    n_alpha, n_beta = ACTIVE_SPACES[state][norb]
    return ActiveSpace(norb, n_alpha, n_beta)


def shift_to_chk_label(shift: str) -> str:
    """Encode a shift string for checkpoint file names: '0.0' -> '0_00', '-0.15' -> 'minus_0_15'."""
    sign = "minus_" if shift.startswith("-") else ""
    integer, fraction = shift.lstrip("-").split(".")
    return f"{sign}{integer}_{fraction.ljust(2, '0')}"


def load_reference(state_dir: Path, shift: str, space: ActiveSpace) -> scf.rohf.ROHF:
    """Rebuild the converged density-fitted ROHF object and attach the AVAS orbitals.

    The molecule, total energy, orbital energies and occupations come from the
    mean-field checkpoint. ``mo_coeff`` is replaced by the AVAS-rotated orbitals
    that define the active space.
    """
    chk_path = state_dir / "chk_files" / f"shift_{shift_to_chk_label(shift)}.chk"
    mo_path = state_dir / "orbitals" / f"shift_{shift}" / f"mo_coeff_{space.label}.npy"

    mol = lib.chkfile.load_mol(str(chk_path))
    mf = scf.ROHF(mol).density_fit()
    mf.__dict__.update(chkfile.load(str(chk_path), "scf"))
    mf.mo_coeff = np.load(mo_path)
    mf.converged = True
    return mf


def frozen_orbitals(mf: scf.rohf.ROHF, space: ActiveSpace) -> list[int]:
    """Indices of the MOs outside the active window [n_core, n_core + norb)."""
    n_mo = mf.mo_coeff.shape[1]
    n_core = (mf.mol.nelectron - space.n_elec) // 2
    active = range(n_core, n_core + space.norb)
    return [i for i in range(n_mo) if i not in active]


def run_uccsd(mf: scf.rohf.ROHF, space: ActiveSpace) -> tuple[float, float, tuple, tuple]:
    """Frozen-core UCCSD and UCCSD(T) in the active space.

    UCCSD is restarted once from its own converged amplitudes to tighten convergence.

    Returns:
        (E_CCSD, E_CCSD(T), t1, t2): total energies in Hartree and the UCCSD
        amplitudes ``t1 = (t1a, t1b)``, ``t2 = (t2aa, t2ab, t2bb)``.
    """
    mycc = cc.UCCSD(mf, frozen=frozen_orbitals(mf, space))
    mycc.kernel()
    mycc.kernel(mycc.t1, mycc.t2)
    e_ccsd = mycc.e_tot
    e_ccsd_t = e_ccsd + mycc.ccsd_t()
    return e_ccsd, e_ccsd_t, mycc.t1, mycc.t2


def t1_diagnostic(t1: tuple[np.ndarray, np.ndarray], n_elec: int) -> float:
    """Open-shell T1 diagnostic: sqrt((|t1a|^2 + |t1b|^2) / n_elec)."""
    t1a, t1b = t1
    return float(np.sqrt((np.sum(t1a**2) + np.sum(t1b**2)) / n_elec))


def save_amplitudes(path: Path, t1: tuple, t2: tuple) -> None:
    """Write UCCSD amplitudes in the format read by the LUCJ circuit builder."""
    np.savez(path, t1a=t1[0], t1b=t1[1], t2aa=t2[0], t2ab=t2[1], t2bb=t2[2])


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")

    space = get_active_space(STATE, NORB)
    state_dir = Path(__file__).resolve().parent / STATE
    log.info("Medium: %s | state: %s | active space: %s (%d qubits)", MEDIUM, STATE, space.label, 2 * space.norb)

    results = {f"{space.label}_{m}": np.zeros(len(SHIFTS)) for m in METHODS}
    t1_diag = np.zeros(len(SHIFTS))

    for i, shift in enumerate(SHIFTS):
        log.info("Shift %s", shift)
        mf = load_reference(state_dir, shift, space)
        e_ccsd, e_ccsd_t, t1, t2 = run_uccsd(mf, space)

        save_amplitudes(state_dir / "orbitals" / f"shift_{shift}" / f"uccsd_amps_{space.label}.npz", t1, t2)
        results[f"{space.label}_ROHF"][i] = mf.e_tot
        results[f"{space.label}_CCSD"][i] = e_ccsd
        results[f"{space.label}_CCSD(T)"][i] = e_ccsd_t
        t1_diag[i] = t1_diagnostic(t1, space.n_elec)

    out_path = state_dir / f"active_space_results_classical_{space.label}.npy"
    np.save(out_path, results, allow_pickle=True)
    log.info("Energies written to %s", out_path)

    header = f"{'shift':>6}  {'E_ROHF':>16}  {'E_CCSD':>16}  {'E_CCSD(T)':>16}  {'T1':>7}"
    rows = [
        f"{s:>6}  {results[space.label + '_ROHF'][i]:16.8f}  {results[space.label + '_CCSD'][i]:16.8f}  "
        f"{results[space.label + '_CCSD(T)'][i]:16.8f}  {t1_diag[i]:7.4f}"
        for i, s in enumerate(SHIFTS)
    ]
    log.info("Summary (Hartree)\n%s\n%s", header, "\n".join(rows))


if __name__ == "__main__":
    main()
