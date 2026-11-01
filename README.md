# QFM: Quantum Frequent-itemset Mining

QFM combines Weighted Bit-Vector Encoding, Mining-Aware Candidate Superposition,
and Bit-Parallel Threshold Marking. This repository provides the QSV threshold
circuit and earlier Amazon Braket validation utilities.

## QSV threshold circuit

`qfm_qsv.py` takes the matching bits of a candidate and the multiplicities of
merged transactions. It includes the fixed offset `K_off` with the weighted
inputs, reduces them with reversible 3-to-2 compressors, and uses a carry-prefix
network to produce the threshold flag. `build_phase_oracle` marks qualifying
states and clears the workspace.

Run the Figure 3 example locally:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install qiskit
python qfm_qsv.py
python -m unittest discover -s tests -v
```

The example uses the paper's candidate `{B, C}`, weights `(3, 3, 3, 2, 2, 2)`,
matching bits `(1, 1, 0, 1, 0, 1)`, and threshold `10`. It checks that `q_th = 1`
and that reversing the circuit clears its workspace.

The module starts with the candidate's matching bits; it does not implement
QPR, candidate preparation, or the QIL listing loop. The example checks the
threshold circuit, rather than reproducing the paper's runtime tables.

## Earlier Braket validation utilities

The following scripts use the earlier sequential support counter. They remain
available for reproducing those validation runs:

- `qfm_paper_toy_qiskit_v29_braket_sweep6_qpu_autopilot.py`: builds and checks
  support-decision circuits on six predefined inputs.
- `qfm_qpu_multirow_braket.py`: samples inputs from UCI files and prepares or
  submits Braket validation runs. Run `--help` for its options.

Install their dependencies with `python -m pip install -r requirements.txt`.
Their polling-based task timing is distinct from the paper's QPU timing model.
AWS credentials are needed for Braket services; QPU submission requires an
explicit `--submit` option in the multirow driver. Local QSV checks above do not
contact a cloud service.
