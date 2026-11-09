# SQD results: [Co(H2O)5CO2]^3+ quintet (gas)

This folder holds the SQD post-processing output for this state, written by [`../main_sqd_post_process.py`](../main_sqd_post_process.py)
(or the matching notebook) when it is run from `sqd_post_processing/gas/`.

Expected layout:

```
<N>qubits/shift_<shift>/
    sqd_history_qubits_<N>_shift_<shift>_<backend>.pkl      per-iteration SCIResult history (git-ignored)
    sqd_convergence_qubits_<N>_shift_<shift>_<backend>.png  energy convergence and orbital occupancies
```

The `.pkl` files are excluded by `.gitignore` because of their size.

*No results are included in this folder yet.*
