#!/usr/bin/env bash
set -uo pipefail

# One-click Table 5 targeted search runner.
# It runs pumsb_star, dense, FIMI-extra, and misc batches sequentially,
# then generates a compact QFM-winning-candidates report.
# It intentionally does NOT use --force, so existing caches in the result dirs are reused.

SCRIPT="${SCRIPT:-run_table_experiments_allinone_optimized_hardened4.py}"
DATA_DIR="${DATA_DIR:-data}"
JAVA_CMD="${JAVA_CMD:-java}"
TIMEOUT_SEC="${TIMEOUT_SEC:-1800}"
BASELINE_JOBS="${BASELINE_JOBS:-1}"
STATS_ALGORITHM="${STATS_ALGORITHM:-FPGrowth_itemsets}"

# Tune these if you want.
JOBS_PUMSB_STAR="${JOBS_PUMSB_STAR:-1}"
JOBS_DENSE="${JOBS_DENSE:-3}"
JOBS_FIMI="${JOBS_FIMI:-3}"
JOBS_MISC="${JOBS_MISC:-4}"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_LOG_DIR="logs_table5_batches_${TS}"
mkdir -p "$RUN_LOG_DIR"

if [ ! -f "$SCRIPT" ]; then
  echo "[error] Cannot find script: $SCRIPT" >&2
  echo "        Set SCRIPT=/path/to/run_table_experiments_allinone_optimized_hardened4.py" >&2
  exit 2
fi

if [ ! -x "$JAVA_CMD" ]; then
  echo "[error] JAVA_CMD not executable: $JAVA_CMD" >&2
  echo "        Set JAVA_CMD=/path/to/java21" >&2
  exit 2
fi

echo "[info] SCRIPT=$SCRIPT"
echo "[info] DATA_DIR=$DATA_DIR"
echo "[info] JAVA_CMD=$JAVA_CMD"
"$JAVA_CMD" -version || true

# ---------------------------------------------------------------------------
# Symlinks for datasets stored in subdirectories.
# ---------------------------------------------------------------------------
mkdir -p "$DATA_DIR"

if [ -f "$DATA_DIR/connect4/connect-4.data" ]; then
  ln -sf connect4/connect-4.data "$DATA_DIR/connect-4.data"
elif [ -f "$DATA_DIR/connect4/connect4.data" ]; then
  ln -sf connect4/connect4.data "$DATA_DIR/connect4.data"
fi

if [ -f "$DATA_DIR/chess/kr-vs-kp.data" ]; then
  ln -sf chess/kr-vs-kp.data "$DATA_DIR/kr-vs-kp.data"
elif [ -f "$DATA_DIR/chess/chess.data" ]; then
  ln -sf chess/chess.data "$DATA_DIR/chess.data"
fi

if [ -f "$DATA_DIR/bike/BIKE.txt" ]; then
  ln -sf bike/BIKE.txt "$DATA_DIR/bike.txt"
fi

KDD_FILE=$(find "$DATA_DIR/kddcup99" -maxdepth 1 -type f \( -name "*.txt" -o -name "*.dat" -o -name "*.data" -o -name "*.spmf" \) 2>/dev/null | head -n 1 || true)
if [ -n "${KDD_FILE:-}" ]; then
  ln -sf "${KDD_FILE#${DATA_DIR}/}" "$DATA_DIR/kddcup99.txt"
fi

# ---------------------------------------------------------------------------
# Helper to run one batch. It logs each batch separately and continues even if
# one batch fails, so you don't lose the remaining runs.
# ---------------------------------------------------------------------------
FAILED=()
run_batch() {
  local name="$1"; shift
  local logfile="$RUN_LOG_DIR/${name}.log"
  echo ""
  echo "===================================================================="
  echo "[batch] $name"
  echo "[log]   $logfile"
  echo "===================================================================="
  "$@" > >(tee "$logfile") 2> >(tee -a "$logfile" >&2)
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "[warn] batch failed: $name rc=$rc" | tee -a "$RUN_LOG_DIR/FAILED.log"
    FAILED+=("$name:$rc")
  else
    echo "[ok] batch finished: $name"
  fi
}

COMMON_ARGS=(
  --mode table5
  --data-dir "$DATA_DIR"
  --timeout-sec "$TIMEOUT_SEC"
  --baseline-jobs "$BASELINE_JOBS"
  --java-cmd "$JAVA_CMD"
  --yu-model paper
  --yu-epsilon 0.01
  --yu-basic-oracle-depth-model lognm
  --yu-candidate-prep-model repeated
  --qfm-start-k 2
  --qfm-preprocess-model nm
  --qfm-cache-update-model logm
  --qfm-popcount-model logm
  --stats-algorithm "$STATS_ALGORITHM"
)

# ---------------------------------------------------------------------------
# Batches. These do not delete existing result dirs and do not pass --force.
# Existing ok/error/timeout caches in the same result dir are reused.
# If you want a clean rerun, manually mv/rm the result directory first.
# ---------------------------------------------------------------------------
run_batch "pumsb_star_target" \
  python "$SCRIPT" \
    "${COMMON_ARGS[@]}" \
    --datasets "pumsb_star" \
    --results-dir results_table5_pumsb_star_target \
    --candidate-minsup "50,40,30,20,10,5,2" \
    --jobs "$JOBS_PUMSB_STAR"

run_batch "dense_candidates" \
  python "$SCRIPT" \
    "${COMMON_ARGS[@]}" \
    --datasets "connect4,chess,nursery" \
    --results-dir results_table5_dense_candidates \
    --candidate-minsup "70,60,50,40,30,20,10,5,2,1,0.5" \
    --jobs "$JOBS_DENSE"

run_batch "fimi_extra" \
  python "$SCRIPT" \
    "${COMMON_ARGS[@]}" \
    --datasets "retail,kosarak,bms1" \
    --results-dir results_table5_fimi_extra \
    --candidate-minsup "20,10,5,2,1,0.5,0.2,0.1" \
    --jobs "$JOBS_FIMI"

run_batch "misc_extra" \
  python "$SCRIPT" \
    "${COMMON_ARGS[@]}" \
    --datasets "bike,kddcup99,recordlink,skin" \
    --results-dir results_table5_misc_extra \
    --candidate-minsup "20,10,5,2,1,0.5,0.2,0.1" \
    --jobs "$JOBS_MISC"

# ---------------------------------------------------------------------------
# Final checker / report.
# ---------------------------------------------------------------------------
REPORT="$RUN_LOG_DIR/qfm_winning_candidates_report.txt"
python - <<'PY' | tee "$REPORT"
import pandas as pd
from pathlib import Path

results_dirs = [
    "results_table5_pumsb_star_target",
    "results_table5_dense_candidates",
    "results_table5_fimi_extra",
    "results_table5_misc_extra",
]

def pick_col(df, names, required=True):
    for n in names:
        if n in df.columns:
            return n
    if required:
        raise KeyError(f"missing columns {names}; available={list(df.columns)}")
    return None

all_ok = []
for RESULTS in results_dirs:
    path = Path(RESULTS) / "table5_candidates.csv"
    print("\n===", RESULTS, "===")
    if not path.exists():
        print("missing:", path)
        continue

    cand = pd.read_csv(path)
    if len(cand) == 0:
        print("empty candidates")
        continue

    qfm_col = pick_col(cand, ["QFM_runtime_sec"])
    yu_col = pick_col(cand, ["Yu_runtime_sec"])
    fp_col = pick_col(cand, ["FP_Growth_runtime_sec", "FPGrowth_itemsets_runtime_sec", "FP-Growth_runtime_sec"], required=False)
    hamm_col = pick_col(cand, ["Hamm_runtime_sec"], required=False)
    eclat_col = pick_col(cand, ["Eclat_runtime_sec"], required=False)

    need = [c for c in [qfm_col, yu_col, fp_col, hamm_col, eclat_col] if c]
    tmp = cand.dropna(subset=need).copy()

    if tmp.empty:
        print("No rows with complete runtime fields.")
        status_cols = [c for c in cand.columns if c.endswith("_status")]
        if status_cols:
            print("Status summary:")
            for c in status_cols:
                print(c, cand[c].value_counts(dropna=False).to_dict())
        continue

    cond = tmp[qfm_col] < tmp[yu_col]
    if fp_col:
        cond &= tmp[qfm_col] < tmp[fp_col]
    if hamm_col:
        cond &= tmp[qfm_col] < tmp[hamm_col]
    if eclat_col:
        cond &= tmp[qfm_col] < tmp[eclat_col]
    ok = tmp[cond].copy()

    cols = [
        "dataset", "minsup_percent", "max_itemset_len", "pattern_count", "L1", "L2",
        qfm_col, fp_col, hamm_col, eclat_col, yu_col,
        "score",
        "FPGrowth_itemsets_status", "FP_Growth_status", "Hamm_status", "Eclat_status", "CICLAD_status",
    ]
    cols = [c for c in cols if c and c in cand.columns]

    if len(ok):
        print("Feasible QFM-winning candidates:")
        shown = ok[cols].sort_values(["dataset", "score"], ascending=[True, False])
        print(shown.to_string(index=False))
        all_ok.append(shown.assign(results_dir=RESULTS))
    else:
        print("No feasible QFM-winning candidates.")
        print("Best candidates by score:")
        if "score" in cand.columns:
            print(cand[cols].sort_values("score", ascending=False).head(15).to_string(index=False))
        else:
            print(cand[cols].head(15).to_string(index=False))

if all_ok:
    merged = pd.concat(all_ok, ignore_index=True)
    out = Path("table5_all_feasible_qfm_winning_candidates.csv")
    merged.to_csv(out, index=False)
    print("\n[done] wrote", out)
else:
    print("\n[done] no feasible QFM-winning candidate found across all batches.")
PY

if [ ${#FAILED[@]} -gt 0 ]; then
  echo ""
  echo "[done-with-warnings] Some batches failed: ${FAILED[*]}"
  echo "See $RUN_LOG_DIR/FAILED.log and per-batch logs."
  exit 1
fi

echo ""
echo "[done] all batches finished."
echo "[report] $REPORT"
echo "[logs]   $RUN_LOG_DIR"
