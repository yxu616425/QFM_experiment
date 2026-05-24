#!/usr/bin/env bash
set -euo pipefail

# Portable rerun script for selected Table-5 points with 24h timeout.
# Does not use `wait -n`, so it works on older bash versions.
# It runs datasets in small batches controlled by PARALLEL_DATASETS.

SCRIPT="${SCRIPT:-run_table_experiments_allinone_optimized_hardened4.py}"
DATA_DIR="${DATA_DIR:-data}"
OUT_ROOT="${OUT_ROOT:-results_table5_24h_timeout_rerun}"
JAVA_CMD="${JAVA_CMD:-java}"
TIMEOUT_SEC="${TIMEOUT_SEC:-86400}"   # 24 hours
PARALLEL_DATASETS="${PARALLEL_DATASETS:-2}"
BASELINE_JOBS="${BASELINE_JOBS:-1}"
STATS_ALGORITHM="${STATS_ALGORITHM:-FPGrowth_itemsets}"
CLASSICAL_METHODS="${CLASSICAL_METHODS:-Eclat,CICLAD}"

if [[ ! -f "$SCRIPT" ]]; then
  echo "[error] SCRIPT not found: $SCRIPT" >&2
  exit 1
fi
if [[ ! -x "$JAVA_CMD" ]]; then
  echo "[error] JAVA_CMD not executable: $JAVA_CMD" >&2
  echo "        Set JAVA_CMD=/path/to/java21" >&2
  exit 1
fi
if ! [[ "$PARALLEL_DATASETS" =~ ^[0-9]+$ ]] || (( PARALLEL_DATASETS < 1 )); then
  echo "[error] PARALLEL_DATASETS must be a positive integer; got: $PARALLEL_DATASETS" >&2
  exit 1
fi

mkdir -p "$OUT_ROOT/logs"

LOG_MAIN="$OUT_ROOT/main.log"
{
  echo "[config] SCRIPT=$SCRIPT"
  echo "[config] DATA_DIR=$DATA_DIR"
  echo "[config] OUT_ROOT=$OUT_ROOT"
  echo "[config] JAVA_CMD=$JAVA_CMD"
  echo "[config] TIMEOUT_SEC=$TIMEOUT_SEC"
  echo "[config] PARALLEL_DATASETS=$PARALLEL_DATASETS"
  echo "[config] BASELINE_JOBS=$BASELINE_JOBS"
  echo "[config] CLASSICAL_METHODS=$CLASSICAL_METHODS"
  "$JAVA_CMD" -version || true
} | tee -a "$LOG_MAIN"

# Root-level aliases for datasets stored in subdirectories.
if [[ -f "$DATA_DIR/connect4/connect-4.data" ]]; then
  ln -sf connect4/connect-4.data "$DATA_DIR/connect-4.data"
elif [[ -f "$DATA_DIR/connect4/connect4.data" ]]; then
  ln -sf connect4/connect4.data "$DATA_DIR/connect4.data"
fi
if [[ -f "$DATA_DIR/chess/kr-vs-kp.data" ]]; then
  ln -sf chess/kr-vs-kp.data "$DATA_DIR/kr-vs-kp.data"
elif [[ -f "$DATA_DIR/chess/chess.data" ]]; then
  ln -sf chess/chess.data "$DATA_DIR/chess.data"
fi

run_one() {
  local ds="$1"
  local ms="$2"
  local outdir="$OUT_ROOT/${ds}_ms${ms}"
  local logfile="$OUT_ROOT/logs/${ds}_ms${ms}.log"

  echo "[start] dataset=$ds minsup=$ms outdir=$outdir" | tee -a "$LOG_MAIN" "$logfile"

  python "$SCRIPT" \
    --mode table5 \
    --datasets "$ds" \
    --data-dir "$DATA_DIR" \
    --results-dir "$outdir" \
    --candidate-minsup "$ms" \
    --timeout-sec "$TIMEOUT_SEC" \
    --jobs 1 \
    --baseline-jobs "$BASELINE_JOBS" \
    --java-cmd "$JAVA_CMD" \
    --classical-methods "$CLASSICAL_METHODS" \
    --yu-model paper \
    --yu-epsilon 0.01 \
    --yu-basic-oracle-depth-model lognm \
    --yu-candidate-prep-model repeated \
    --qfm-start-k 2 \
    --qfm-preprocess-model nm \
    --qfm-cache-update-model logm \
    --qfm-popcount-model logm \
    --stats-algorithm "$STATS_ALGORITHM" \
    ${FORCE_FLAG:-} \
    >> "$logfile" 2>&1

  echo "[done] dataset=$ds minsup=$ms" | tee -a "$LOG_MAIN" "$logfile"
}

DATASETS=("accidents" "connect4" "pumsb" "pumsb_star" "chess")
MINSUPS=("10"        "70"       "60"    "30"         "70")

pids=()
for i in "${!DATASETS[@]}"; do
  run_one "${DATASETS[$i]}" "${MINSUPS[$i]}" &
  pids+=("$!")

  if (( ${#pids[@]} >= PARALLEL_DATASETS )); then
    echo "[batch] waiting for ${#pids[@]} job(s): ${pids[*]}" | tee -a "$LOG_MAIN"
    failed=0
    for pid in "${pids[@]}"; do
      if ! wait "$pid"; then
        echo "[warn] child job failed: pid=$pid" | tee -a "$LOG_MAIN"
        failed=1
      fi
    done
    pids=()
    if (( failed != 0 )); then
      echo "[warn] one or more jobs in this batch failed; continuing to next batch" | tee -a "$LOG_MAIN"
    fi
  fi
done

if (( ${#pids[@]} > 0 )); then
  echo "[batch] waiting for final ${#pids[@]} job(s): ${pids[*]}" | tee -a "$LOG_MAIN"
  failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      echo "[warn] child job failed: pid=$pid" | tee -a "$LOG_MAIN"
      failed=1
    fi
  done
  if (( failed != 0 )); then
    echo "[warn] one or more final jobs failed; summary will still be generated" | tee -a "$LOG_MAIN"
  fi
fi

echo "[phase] Summarizing 24h rerun results ..." | tee -a "$LOG_MAIN"
OUT_ROOT_ENV="$OUT_ROOT" python - <<'PY'
import os
import pandas as pd
from pathlib import Path

OUT_ROOT = Path(os.environ["OUT_ROOT_ENV"])
pairs = [
    ("accidents", "10"),
    ("connect4", "70"),
    ("pumsb", "60"),
    ("pumsb_star", "30"),
    ("chess", "70"),
]
rows = []
for ds, ms in pairs:
    d = OUT_ROOT / f"{ds}_ms{ms}"
    cand_path = d / "table5_candidates.csv"
    row = {"dataset": ds, "minsup_percent": float(ms), "results_dir": str(d)}
    if not cand_path.exists():
        row["status"] = "missing_table5_candidates"
        rows.append(row)
        continue
    try:
        cand = pd.read_csv(cand_path)
    except Exception as e:
        row["status"] = "read_error"
        row["error"] = str(e)
        rows.append(row)
        continue
    if len(cand) == 0:
        row["status"] = "empty_candidates"
        rows.append(row)
        continue
    r = cand.iloc[0].to_dict()
    for c in [
        "QFM_runtime_sec", "Yu_runtime_sec", "FP_Growth_runtime_sec", "FPGrowth_itemsets_runtime_sec",
        "Hamm_runtime_sec", "Eclat_runtime_sec", "CICLAD_runtime_sec",
        "FP_Growth_status", "FPGrowth_itemsets_status", "Hamm_status", "Eclat_status", "CICLAD_status",
        "max_itemset_len", "pattern_count", "L1", "L2", "score"
    ]:
        if c in r:
            row[c] = r[c]
    row["status"] = "ok"
    rows.append(row)

out = pd.DataFrame(rows)
summary_csv = OUT_ROOT / "timeout_rerun_summary.csv"
out.to_csv(summary_csv, index=False)
print(f"[ok] wrote {summary_csv}")
print(out.to_string(index=False))

report = OUT_ROOT / "timeout_rerun_report.txt"
with report.open("w") as f:
    f.write("24h timeout rerun summary\n")
    f.write("==========================\n\n")
    f.write(out.to_string(index=False))
    f.write("\n")
print(f"[ok] wrote {report}")
PY

echo "[done] Summary: $OUT_ROOT/timeout_rerun_summary.csv" | tee -a "$LOG_MAIN"
echo "[done] Report:  $OUT_ROOT/timeout_rerun_report.txt" | tee -a "$LOG_MAIN"
