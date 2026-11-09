# Sampler results: [Co(H2O)5CO2]^2+ doublet (gas)

This folder holds the measured bitstrings from the IBM Quantum hardware runs for this state. It is filled by
[`../run_saved_circuits.ipynb`](../run_saved_circuits.ipynb) and read by
`../../../sqd_post_processing/gas/`.

Expected layout, one file per geometry shift (-0.15, 0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2):

```
<N>qubits/qubits_<N>_shift_<shift>_<backend>.json
```

Each file is a single `SamplerPubResult` serialised with `qiskit_ibm_runtime.RuntimeEncoder`.
Load it with `json.load(f, cls=RuntimeDecoder)`; the bitstrings are in `pub_result.data.meas`.

| N (qubits) | active space (norb, nα, nβ) |
|---|---|
| 34 | (17, 12, 11) |

*No results are included in this folder yet.*
