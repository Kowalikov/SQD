# Fire Opal circuits: [Co(H2O)5CO2]^3+ singlet (solvent)

This folder holds the circuits produced by Q-CTRL Fire Opal, which takes the circuits in
`../transpiler_pass_initial/` as input. They are used by `../../../../analysis/QC_EDA.ipynb`
to compare circuit metrics (SI Figures S2/S3).

Expected file name, one per geometry shift (-0.15, 0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2) and qubit count (34, 38, 46, 50):

```
qubits_<N>_shift_<shift>_<backend>_no_frac_gates.qpy
```

*No circuits are included in this folder yet.*
