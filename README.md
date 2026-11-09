# SQD for open-shell transition-metal complexes in gas and implicit solvent

Code and data accompanying

> D. David, V. Pathak, M. Kowalik, H. Mohammadbagherpoor, V. Beltrani, K. Maller, N. Moroney, P. Lolur,
> *Sample-based quantum diagonalization approach for open-shell transition-metal complexes in gas and implicit-solvent*,
> arXiv:2607.16389 (2026). [doi:10.48550/arXiv.2607.16389](https://doi.org/10.48550/arXiv.2607.16389)

The repository covers the full workflow used in the paper. It applies open-shell sample-based quantum diagonalization (SQD) to the octahedral complex [Co(H2O)5CO2]^{2+/3+}:
- in the gas phase and in implicit solvent (IEF-PCM, through an outer self-consistent reaction-field loop around SQD);
- along a Co–O(CO2) dissociation path;
- for the Co(III) singlet/quintet and the Co(II) doublet/quartet;
- with active spaces of up to 50 qubits, sampled on IBM Heron hardware.

## States and active spaces

| State | Folder | Active spaces `(norb, nα, nβ)` | Qubits |
|---|---|---|---|
| Co(III) singlet, [Co(H2O)5CO2]^3+ | `charge3_spin0` | (17,11,11), (19,12,12), (23,14,14), (25,15,15) | 34, 38, 46, 50 |
| Co(III) quintet, [Co(H2O)5CO2]^3+ | `charge3_spin4` | (17,13,9), (19,14,10), (23,16,12), (25,17,13) | 34, 38, 46, 50 |
| Co(II) doublet, [Co(H2O)5CO2]^2+ | `charge2_spin1` | (17,12,11) | 34 |
| Co(II) quartet, [Co(H2O)5CO2]^2+ | `charge2_spin3` | (17,13,10) | 34 |

Each state is computed at 10 geometries. The geometry is labelled by `shift`:

| shift | -0.15 | 0.0 | 0.15 | 0.3 | 0.45 | 0.6 | 0.75 | 0.9 | 1.05 | 1.2 |
|---|---|---|---|---|---|---|---|---|---|---|
| Co–O(CO2) [Å] | 1.44 | 1.97 | 2.51 | 3.04 | 3.57 | 4.10 | 4.63 | 5.16 | 5.69 | 6.23 |

## Repository layout and workflow

```
data_Co_5_H2O_CO2/        1. geometries (.xyz) and the notebook that generates the dissociation path
classical_methods/        2. mean-field checkpoints, AVAS orbitals, FCIDUMPs, UCCSD amplitudes, ROHF/(U)CCSD(T) energies
quantum_circuits/         3. LUCJ circuits built from the UCCSD amplitudes, and their transpiled versions (QPY)
sampler_runs/             4. hardware sampling of the transpiled circuits (SamplerV2)
sqd_post_processing/      5. SQD (gas) and SQD + SCRF/IEF-PCM (solvent) post-processing of the samples
analysis/                 6. notebooks and data for the paper figures
```

Every stage is split into `{gas,solvent}/charge{C}_spin{S}/`. Stages exchange data only through files at fixed relative paths:

```
data_Co_5_H2O_CO2/structure/*.xyz
        │
        ▼
classical_methods/<medium>/<state>/
    chk_files/  orbitals/shift_*/{mo_coeff,uccsd_amps}_*  active_spaces/  active_space_results_classical_*.npy
        │                                   │
        │ uccsd_amps                         │ chk + mo_coeff (integrals), CCSD reference
        ▼                                   │
quantum_circuits/<medium>/build_circuits*.ipynb
    <state>/lucj_circuits/  <state>/transpiler_pass_initial/<N>_qubits/
        │                                   │
        ▼                                   │
sampler_runs/<medium>/run_saved_circuits.ipynb
    <state>/<N>qubits/*.json                │
        │                                   │
        ▼                                   ▼
sqd_post_processing/<medium>/main_sqd_post_process*.py
    <state>/<N>qubits/shift_<shift>/
```

### What is included

| Stage | Contents |
|---|---|
| `data_Co_5_H2O_CO2/` | All 11 geometries, and the notebook that regenerates them |
| `classical_methods/` | Complete data for all states, active spaces and geometries, plus one `classical_calculations.py` per medium for the coupled-cluster benchmarks ([details](classical_methods/README.md)) |
| `quantum_circuits/` | Generation notebooks, LUCJ circuits (`lucj_circuits/`) and transpiled circuits (`transpiler_pass_initial/`) for all states and active spaces, for `ibm_pittsburgh` |
| `quantum_circuits/*/*/fire_opal_circuits/` | Placeholder for the Q-CTRL Fire Opal circuits |
| `sampler_runs/` | Sampling notebooks. Per-state result folders are placeholders; see the README in each |
| `sqd_post_processing/` | SQD scripts and notebooks. Per-state output folders are placeholders; see the README in each |
| `analysis/` | Figure notebooks, the energies behind Figures 2–4 and 9–15 as CSV, and the published figures ([details](analysis/README.md)) |

## Installation

```bash
conda create -n sqd-co python=3.13
conda activate sqd-co
pip install -r requirements.txt
```

The paper's SI reports Qiskit 2.2.1 for the transpilation step. `requirements.txt` pins the environment that was used to run and test the notebooks and scripts in this repository.

Hardware access is needed only for building the transpiled circuits (backend calibration data) and for sampling. It requires an IBM Quantum account saved with `QiskitRuntimeService.save_account(...)`; the code calls `QiskitRuntimeService()` without credentials.
`run_saved_circuits.ipynb` has `run_sampler=False` by default. Setting it to `True` submits a job and consumes QPU time.

## Running

Scripts and notebooks take no command-line arguments. They are configured by the variables at the top (`charge`, `spin_number`, `medium`, `shift`, `num_orbitals`, `num_elec_a`, `num_elec_b`, `backend_name`, and the SQD hyper-parameters).
Paths are relative. Run the SQD scripts and the notebooks from their own folder; `classical_calculations.py` resolves its paths from its own location and can be run from anywhere:

```bash
# coupled-cluster benchmarks in the active space
python classical_methods/gas/classical_calculations.py      # set STATE and NORB at the top of the script first

# SQD post-processing (needs the sampler results for the chosen state in sampler_runs/)
cd sqd_post_processing/gas && python main_sqd_post_process.py
cd sqd_post_processing/solvent && python main_sqd_post_process_solvent.py

# figures
cd analysis && jupyter notebook
```

## File-naming conventions

- `as_label = norb_<norb>_nelec_<nα>_<nβ>` (e.g. `norb_17_nelec_13_9`) identifies the active space.
- Folder and file names use the shift string as is (`shift_0.3`, `qubits_34_shift_0.3_*.qpy`).
- Mean-field checkpoints use an encoded shift instead: `0.0 → shift_0_00.chk`, `0.3 → shift_0_30.chk`, `-0.15 → shift_minus_0_15.chk`.
- `spin_number` is 2S. Spin symmetrisation in SQD is enabled only for the singlet (`symmetrize_spin = not bool(spin_number)`).

## Citation

Please cite the article above if you use this code or data. Citation metadata is in [`CITATION.cff`](CITATION.cff).

## License

[Apache License 2.0](LICENSE).
