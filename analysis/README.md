# Analysis and paper figures

Notebooks that make the figures in the paper. Run them from this folder; they write to `./figures/`.
The figures exactly as published are in `./paper_figures/Figure_<n>.pdf`, numbered as in the paper.

| Paper figure | Content | Notebook | Data |
|---|---|---|---|
| 1 | Workflow overview (schematic) | n/a, shipped as `paper_figures/Figure_1.pdf` | n/a |
| 2 | Gas, charge 3+: equilibrium energies vs. qubits | `Results_clean_charge_3.ipynb` → `gas_charge_3_eq_point_results.pdf` | `data/energies_charge3.csv` |
| 3 | Gas, charge 3+: potential energy curves | `Results_clean_charge_3.ipynb` → `gas_charge_3_results.pdf` | `data/energies_charge3.csv` |
| 4 | Gas, charge 3+: singlet–quintet gap | `Results_clean_charge_3.ipynb` → `gas_charge_3_diff_results.pdf` | `data/energies_charge3.csv` |
| 5 | τ1 diagnostics, charge 3+ | `t1_t2_diagnostic.ipynb` → `Figure_5.pdf` | `data/t1_diagnostics_Co_CO2_5H2O.txt` |
| 6 | τ1 diagnostics, charge 2+ | `t1_t2_diagnostic.ipynb` → `Figure_6.pdf` | `data/t1_diagnostics_Co_CO2_5H2O.txt` |
| 7 | Mulliken populations, charge 3+ | `t1_t2_diagnostic.ipynb` → `Figure_7.pdf` | `data/mulliken_results_with_IBM_scf_checkfiles.txt` |
| 8 | Mulliken populations, charge 2+ | `t1_t2_diagnostic.ipynb` → `Figure_8.pdf` | `data/mulliken_results_with_IBM_scf_checkfiles.txt` |
| 9 | Gas, charge 2+: potential energy curves | `Results_clean_charge_2.ipynb` → `gas_charge_2_results.pdf` | `data/energies_charge2.csv` |
| 10 | Gas, charge 2+: doublet–quartet gap | `Results_clean_charge_2.ipynb` → `gas_charge_2_diff_results.pdf` | `data/energies_charge2.csv` |
| 11 | Solvent, charge 2+: potential energy curves | `Results_clean_charge_2.ipynb` → `solvent_charge_2_results.pdf` | `data/energies_charge2.csv` (see note) |
| 12 | Solvent, charge 2+: doublet–quartet gap | `Results_clean_charge_2.ipynb` → `solvent_charge_2_diff_results.pdf` | `data/energies_charge2.csv` (see note) |
| 13 | Solvent, charge 3+: potential energy curves | `Results_clean_charge_3.ipynb` → `solvent_charge_3_results.pdf` | `data/energies_charge3.csv` |
| 14 | Solvent, charge 3+: singlet–quintet gap | `Results_clean_charge_3.ipynb` → `solvent_charge_3_diff_results.pdf` | `data/energies_charge3.csv` |
| 15 | Solvent, charge 3+: equilibrium energies vs. qubits | `Results_clean_charge_3.ipynb` → `solvent_charge_3_eq_point_results.pdf` | `data/energies_charge3.csv` |
| S2 | Circuit metrics, charge 3+ (default vs. Fire Opal transpilation) | `QC_EDA.ipynb` → `qc_metrics_charge3_{gas,solvent}.png` | tabulated in the notebook; recomputed from `../quantum_circuits/` with `RUN_FROM_SCRATCH = True` |

**Note on Figures 11 and 12.** `data/energies_charge2.csv` does not include the solvent-phase HCI energies for charge 2+.
For the figures to show the HCI curves, add rows with `medium=solvent, method=HCI, n_qubits=34` for the doublet and quartet. The notebook draws them automatically when they are present.
The ROHF, UCCSD(T) and SQD curves are reproduced from the CSV.

## Data files

### `data/energies_charge{2,3}.csv`
Total energies in Hartree, in long format. There is one row per (medium, state, method, active space, geometry).

| Column | Meaning |
|---|---|
| `medium` | `gas` or `solvent` (IEF-PCM) |
| `charge` | 2 or 3 |
| `state` | `singlet`, `quintet` (charge 3) / `doublet`, `quartet` (charge 2) |
| `method` | `ROHF` (RHF for the singlet), `CCSD(T)` (UCCSD(T) for open shells), `HCI`, `SQD` |
| `n_qubits` | active-space size in qubits (`2 × norb`); empty for the mean-field energies |
| `shift` | geometry shift label (see `../data_Co_5_H2O_CO2/`) |
| `distance_angstrom` | Co–O(CO2) distance |
| `energy_hartree` | total energy |

The ROHF/RHF and (U)CCSD(T) entries match `active_space_results_classical_<as_label>.npy` in `../classical_methods/`.

### `data/t1_diagnostics_Co_CO2_5H2O.txt`, `data/mulliken_results_with_IBM_scf_checkfiles.txt`
Plain-text tables, with a column legend at the end of each file.
- τ1 diagnostics come from (U)CCSD in the active space: (30e, 25o) for charge 3+ and (23e, 17o) for charge 2+.
- Gas-phase Mulliken charges of the CO2 and Co(H2O)5 fragments come from the (RO)HF one-particle density matrices.

## Requirements
`QC_EDA.ipynb` uses the `science` style from SciencePlots. It renders with LaTeX when a LaTeX installation is found, and falls back to matplotlib mathtext otherwise.
