# QFM Experiment Driver

This repository contains the experiment scripts used to generate/inspect QFM comparison tables.

## What is included

- `run_table_experiments_allinone.py`  
  Main all-in-one driver.
  - `--mode table12`: Table 1/2 QFM vs Yu/qARM full-level quantum-proxy sweeps.
  - `--mode table5`: Table 5 large-dataset comparison with FP-Growth, Eclat, Hamm, and CICLAD.
  - `--mode all`: run both.
- `inspect_results.py`  
  Summarizes generated CSV outputs.
- `merge_table5_results.py`  
  Merges multiple Table 5 result directories and selects QFM-winning candidates when available.
- `scripts/run_table5_target_batches.sh`  
  Convenience script for targeted Table 5 dataset batches.
- `scripts/run_table5_24h_timeout_rerun_portable.sh`  
  Convenience script for re-running selected long-running methods with a 24-hour timeout.

## What is not included

This repository intentionally does **not** include datasets, large result directories, or third-party binaries/jars. You need to provide:

- datasets under `data/` or `data_raw/`;
- SPMF jar, default path `tools/spmf.jar`;
- Hamm executable, default path `tools/hamm` or configure via `--hamm-bin` / `HAMM_BIN`;
- CICLAD executable, default path `tools/ciclad` or configure via `--ciclad-bin` / `CICLAD_BIN`.

SPMF currently requires Java 21 if your `spmf.jar` was compiled for class-file version 65. Use `--java-cmd /path/to/java21` or set `JAVA_CMD`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place external tools as:

```text
tools/spmf.jar
tools/ciclad
tools/hamm
```

or pass explicit paths:

```bash
--spmf-jar /path/to/spmf.jar \
--java-cmd /path/to/java21 \
--hamm-bin /path/to/hamm \
--ciclad-bin /path/to/ciclad
```

## Example: Table 1/2 QFM vs Yu sweeps

```bash
python run_table_experiments_allinone.py \
  --mode table12 \
  --table12-datasets "mushroom,connect4,chess,tic_tac_toe,car,nursery" \
  --table12-data-dir data_raw \
  --results-dir results_table12 \
  --tx-ratios "10,20,30,40,50,60,70" \
  --minsup-ratios "10,20,30,40,50,60,70" \
  --override-default-minsup "mushroom=10,connect4=10,tic-tac-toe=10,car=10,kr-vs-kp=10,nursery=10" \
  --tx-sweep-minsup-mode count \
  --table12-jobs 4 \
  --stats-algorithm FPGrowth_itemsets \
  --yu-model paper \
  --yu-epsilon 0.01 \
  --yu-basic-oracle-depth-model lognm \
  --yu-candidate-prep-model repeated \
  --qfm-start-k 2 \
  --qfm-preprocess-model nm \
  --qfm-cache-update-model logm \
  --qfm-popcount-model logm
```

Outputs:

```text
results_table12/table1_tx_sweep_quantum_full.csv
results_table12/table2_minsup_sweep_quantum_full.csv
```

## Example: Table 5 targeted large-dataset comparison

```bash
python run_table_experiments_allinone.py \
  --mode table5 \
  --datasets "accidents,connect4,pumsb,pumsb_star,chess" \
  --data-dir data \
  --results-dir results_table5 \
  --candidate-minsup "70,60,50,40,30,20,10,5,2,1,0.5" \
  --timeout-sec 1800 \
  --jobs 3 \
  --baseline-jobs 1 \
  --stats-algorithm FPGrowth_itemsets \
  --java-cmd "$JAVA_CMD" \
  --yu-model paper \
  --yu-epsilon 0.01 \
  --yu-basic-oracle-depth-model lognm \
  --yu-candidate-prep-model repeated \
  --qfm-start-k 2 \
  --qfm-preprocess-model nm \
  --qfm-cache-update-model logm \
  --qfm-popcount-model logm
```

Outputs:

```text
results_table5/table5_candidates.csv
results_table5/table5_selected_minsup.csv
results_table5/table5_final.csv
```

## Inspect results

```bash
python inspect_results.py --results-dir results_table5
```

This writes summaries under:

```text
results_table5/analysis_summary/
```

## Merge Table 5 results

```bash
python merge_table5_results.py \
  --base-results results_hardened4_all \
  --extra-results "results_table5_accidents_target,results_table5_dense_candidates,results_table5_fimi_extra,results_table5_misc_extra" \
  --output-dir results_table5_merged_all \
  --require-win-against "FP-Growth,Hamm,Eclat,Yu"
```

## 24-hour rerun for timeout methods

```bash
TIMEOUT_SEC=86400 PARALLEL_DATASETS=2 \
  scripts/run_table5_24h_timeout_rerun_portable.sh
```

Environment variables accepted by the shell scripts:

- `JAVA_CMD` path to Java 21 executable, default `java`;
- `SCRIPT` main Python script path, default `run_table_experiments_allinone.py`;
- `DATA_DIR`, default `data`;
- `TIMEOUT_SEC`, default `86400` for 24-hour rerun;
- `PARALLEL_DATASETS`, default `2`.

## Notes on interpretation

- QFM and Yu runtimes are analytical depth proxies converted by `gate_time_ns`.
- Classical baseline runtimes are wall-clock subprocess measurements.
- Timeout/pruned results should be reported transparently. `>24h` means the method did not finish within the 24-hour cutoff.
- If you use QFM-favorable support selection, keep `table5_candidates.csv` as an audit trail.
