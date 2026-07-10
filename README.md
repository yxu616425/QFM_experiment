# QFM — Quantum Frequent-itemset Mining (core implementation)

Reference implementation of the QFM candidate-verification circuits
(bit-vector qubit encoding and the threshold-marking oracle), together
with the scripts used for the physical-QPU and simulated-noise validation.

## Contents

- `qfm_paper_toy_qiskit_v29_braket_sweep6_qpu_autopilot.py`
  Builds the QFM threshold-oracle circuits for the sampled datasets and
  runs them on Qiskit Aer or on Amazon Braket devices.
- `qfm_qpu_multirow_braket.py`
  One-click physical-QPU validation driver (SV1 dry-run validation gate,
  then QPU submission). Auto-locates the circuit builder above; see
  `--help` for options.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

AWS credentials are required only for Amazon Braket QPU submissions;
all circuits can also be executed locally on the Aer simulator.
