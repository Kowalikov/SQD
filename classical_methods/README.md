# Classical methods: mean-field references, AVAS active spaces and coupled-cluster benchmarks

This folder holds the classical quantum-chemistry inputs for every later stage, plus the ROHF / (U)CCSD / (U)CCSD(T)
benchmark energies in each active space. Each medium has one benchmark script and one data folder per state:

```
gas/
    classical_calculations.py             (U)CCSD / (U)CCSD(T) benchmarks, gas phase
    charge{C}_spin{S}/                    data for one state; S = 2S (number of unpaired electrons)
solvent/
    classical_calculations.py             same, using the IEF-PCM mean-field references
    charge{C}_spin{S}/
```

| Folder | State | Active spaces `(norb, nα, nβ)` | Qubits |
|---|---|---|---|
| `charge3_spin0` | [Co(H2O)5CO2]^3+ singlet | (17,11,11), (19,12,12), (23,14,14), (25,15,15) | 34, 38, 46, 50 |
| `charge3_spin4` | [Co(H2O)5CO2]^3+ quintet | (17,13,9), (19,14,10), (23,16,12), (25,17,13) | 34, 38, 46, 50 |
| `charge2_spin1` | [Co(H2O)5CO2]^2+ doublet | (17,12,11) | 34 |
| `charge2_spin3` | [Co(H2O)5CO2]^2+ quartet | (17,13,10) | 34 |

All data covers the 10 geometries along the CO2 dissociation path in [`../data_Co_5_H2O_CO2/structure/`](../data_Co_5_H2O_CO2/structure/).
The shifts are `-0.15, 0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2`, which correspond to Co–O(CO2) distances of 1.44 … 6.23 Å.

## How the data were generated

The paper's Methods section (Sections 2.2, 2.3 and 2.9) and SI Section S1 describe the procedure in full. In short:

- **Mean field.**
  - Basis: def2-TZVP, with density fitting using the PySCF def2-TZVP auxiliary basis.
  - The singlet uses RHF; the doublet, quartet and quintet use ROHF.
  - Starting density: a fragment-based guess, with Co treated as open-shell and the ligands as closed-shell.
  - Convergence: second-order SCF solver, threshold 1e-9. Internal stability analysis was run, and the density was re-optimised where needed.
  - Along the path, the converged density matrix was propagated between adjacent geometries.
  - Solvent: IEF-PCM with ε = 78.3553.
- **Active space.**
  - Method: AVAS projection onto Co 3d, 4s, 4p and the CO2 C 2s, 2p and O 2s, 2p atomic orbitals.
  - Separate occupied and virtual thresholds give the sizes in the table above.
- **Benchmarks.**
  - ROHF, UCCSD and UCCSD(T) energies are computed in the frozen-core active space by [`gas/classical_calculations.py`](gas/classical_calculations.py) and [`solvent/classical_calculations.py`](solvent/classical_calculations.py).
  - Solvent runs use the PCM-polarised ROHF reference.

## Files in each `charge{C}_spin{S}/` folder

`<shift>` is the shift string (`0.3`). `<shift_core>` is the encoded form used for checkpoint names (`0.0 → 0_00`, `0.3 → 0_30`, `-0.15 → minus_0_15`).
`<as_label> = norb_<norb>_nelec_<nα>_<nβ>`.

| Path | Content |
|---|---|
| `chk_files/shift_<shift_core>.chk` | PySCF checkpoint of the converged mean-field solution: `mol`, `scf/{e_tot, mo_coeff, mo_energy, mo_occ}` |
| `orbitals/shift_<shift>/mo_coeff_<as_label>.npy` | AVAS-rotated MO coefficients (n_AO × n_MO) defining the active space |
| `orbitals/shift_<shift>/uccsd_amps_<as_label>.npz` | UCCSD amplitudes `t1a, t1b, t2aa, t2ab, t2bb` (active space only), used to build the LUCJ circuits |
| `active_spaces/shift_<shift>/spin_<S>_shift_<shift>_<as_label>.fcidump` | Active-space Hamiltonian (h1e, h2e, E_core) in FCIDUMP format |
| `active_space_results_classical_<as_label>.npy` | Pickled dict `{"<as_label>_ROHF" \| "_CCSD" \| "_CCSD(T)": array of 10 energies (Ha), one per shift}`; load with `np.load(..., allow_pickle=True).item()` |
| `initial_rdm*.npy` | Fragment-based initial density matrix used to start the SCF |
| `evals_ansatz*.npy` | Mean-field total energies (Ha) along the 10 shifts, obtained from the corresponding initial guess |

## Re-running the coupled-cluster benchmarks

Set `STATE` (e.g. `"charge3_spin4"`) and `NORB` (one of the active-space sizes in the table above) at the top of the script, then run:

```bash
python gas/classical_calculations.py        # or: python solvent/classical_calculations.py
```

Paths are resolved relative to the script, so you can run it from any directory.
For each of the 10 shifts it logs the ROHF, UCCSD and UCCSD(T) energies, and it ends with a summary table that includes the T1 diagnostic.

> **Note:** the script overwrites `<STATE>/orbitals/shift_*/uccsd_amps_<as_label>.npz` and `<STATE>/active_space_results_classical_<as_label>.npy`.
