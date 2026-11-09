# SQD results: [Co(H2O)5CO2]^2+ quartet (solvent)

This folder holds the SQD post-processing output for this state, written by [`../main_sqd_post_process_solvent.py`](../main_sqd_post_process_solvent.py)
(or the matching notebook) when it is run from `sqd_post_processing/solvent/`.

Expected layout:

```
<N>qubits/shift_<shift>/
    sqd_solvent_scrf_<N>q_<backend>.pkl                  full SCRF + inner S-CORE history (git-ignored)
    sqd_solvent_summary_<N>q_<backend>.json              per-SCRF-iteration energies (E_active, E_core, E_total, E_solvent, double-counting term)
    sqd_solvent_convergence_<N>q_<backend>.png           SCRF convergence
    sqd_solvent_inner_convergence_<N>q_<backend>.png     inner S-CORE convergence per SCRF cycle
```

The `.pkl` files are excluded by `.gitignore` because of their size.

*No results are included in this folder yet.*
