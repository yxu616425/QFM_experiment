#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inspect and summarize hardened4 / hardened4_fpgrowth experiment outputs.

Usage:
  python inspect_hardened4_results.py --results-dir results_hardened4_fpgrowth_all

It reads, when present:
  - table1_tx_sweep_quantum_full.csv
  - table2_minsup_sweep_quantum_full.csv
  - table5_final.csv
  - table5_selected_minsup.csv
  - table5_candidates.csv

It writes compact summary files under <results-dir>/analysis_summary/:
  - summary_report.md
  - table1_qfm_yu_ratio.csv
  - table2_qfm_yu_ratio.csv
  - table5_runtime_pivot.csv
  - table5_status_summary.csv
  - table5_candidates_compact.csv
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import pandas as pd


def read_csv_or_none(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[warn] failed to read {path}: {e}")
        return None


def pick_cols(df: pd.DataFrame, cols: Sequence[str]) -> List[str]:
    return [c for c in cols if c in df.columns]


def safe_write_csv(df: Optional[pd.DataFrame], path: Path) -> None:
    if df is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[ok] wrote {path} ({len(df)} rows)")


def fmt_float(x, digits: int = 6) -> str:
    try:
        if pd.isna(x):
            return "NA"
        x = float(x)
        if abs(x) >= 1e6 or (abs(x) > 0 and abs(x) < 1e-4):
            return f"{x:.{digits}e}"
        return f"{x:.{digits}g}"
    except Exception:
        return str(x)


def df_to_md(df: Optional[pd.DataFrame], max_rows: int = 30) -> str:
    if df is None:
        return "_missing_\n"
    if len(df) == 0:
        return "_empty_\n"
    tmp = df.copy()
    if len(tmp) > max_rows:
        tmp = tmp.head(max_rows)
        suffix = f"\n\n_Showing first {max_rows} rows of {len(df)}._\n"
    else:
        suffix = "\n"
    # Avoid very long floats in markdown.
    for c in tmp.columns:
        if pd.api.types.is_float_dtype(tmp[c]):
            tmp[c] = tmp[c].map(lambda v: fmt_float(v))
    try:
        return tmp.to_markdown(index=False) + suffix
    except Exception:
        return tmp.to_string(index=False) + suffix


def qfm_yu_ratio(df: Optional[pd.DataFrame], kind: str) -> Optional[pd.DataFrame]:
    if df is None or df.empty or "method" not in df.columns or "runtime_sec" not in df.columns:
        return None
    idx = ["dataset"]
    if kind == "tx" and "tx_ratio_percent" in df.columns:
        idx.append("tx_ratio_percent")
    if "minsup_percent" in df.columns:
        idx.append("minsup_percent")
    if "effective_minsup_percent" in df.columns and kind == "tx":
        idx.append("effective_minsup_percent")
    # Drop duplicated index cols while preserving order.
    idx = list(dict.fromkeys(idx))
    try:
        p = df.pivot_table(index=idx, columns="method", values="runtime_sec", aggfunc="first").reset_index()
    except Exception as e:
        print(f"[warn] failed pivot for {kind}: {e}")
        return None
    if "QFM" in p.columns and "Yu" in p.columns:
        p["QFM_over_Yu"] = p["QFM"] / p["Yu"]
        p["QFM_faster_than_Yu"] = p["QFM_over_Yu"] < 1.0
    return p


def table5_pivot(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if df is None or df.empty or "method" not in df.columns or "runtime_sec" not in df.columns:
        return None
    idx = ["dataset"]
    if "minsup_percent" in df.columns:
        idx.append("minsup_percent")
    try:
        p = df.pivot_table(index=idx, columns="method", values="runtime_sec", aggfunc="first").reset_index()
    except Exception as e:
        print(f"[warn] failed table5 pivot: {e}")
        return None
    if "QFM" in p.columns:
        for method in [c for c in p.columns if c not in set(idx + ["QFM"] )]:
            if pd.api.types.is_numeric_dtype(p[method]):
                p[f"QFM_over_{method}"] = p["QFM"] / p[method]
    return p


def status_summary(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None
    cols = [c for c in ["dataset", "method", "status"] if c in df.columns]
    if not cols or "status" not in cols:
        return None
    out = df.groupby(cols, dropna=False).size().reset_index(name="count")
    return out.sort_values(cols).reset_index(drop=True)


def compact_candidates(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None
    preferred = [
        "dataset", "minsup_percent", "method", "runtime_sec", "status",
        "pattern_count", "max_itemset_len", "l1_count", "l2_count",
        "qfm_runtime_sec", "yu_runtime_sec", "selection_score",
        "qfm_vs_yu_ratio", "qfm_vs_median_classical_ratio", "qfm_vs_fastest_classical_ratio",
        "selection_reason", "cmd", "error",
    ]
    cols = pick_cols(df, preferred)
    if not cols:
        return df.head(100)
    return df[cols].copy()


def method_list(df: Optional[pd.DataFrame]) -> List[str]:
    if df is None or "method" not in df.columns:
        return []
    return sorted([str(x) for x in df["method"].dropna().unique()])


def dataset_list(df: Optional[pd.DataFrame]) -> List[str]:
    if df is None or "dataset" not in df.columns:
        return []
    return sorted([str(x) for x in df["dataset"].dropna().unique()])


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize hardened4/hardened4_fpgrowth experiment outputs.")
    ap.add_argument("--results-dir", required=True, help="Experiment results directory, e.g. results_hardened4_fpgrowth_all")
    ap.add_argument("--out-dir", default=None, help="Output directory. Default: <results-dir>/analysis_summary")
    ap.add_argument("--max-report-rows", type=int, default=40, help="Max rows to show per table in summary_report.md")
    args = ap.parse_args()

    results = Path(args.results_dir)
    if not results.exists():
        raise SystemExit(f"[error] results dir not found: {results}")
    out_dir = Path(args.out_dir) if args.out_dir else results / "analysis_summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    table1 = read_csv_or_none(results / "table1_tx_sweep_quantum_full.csv")
    table2 = read_csv_or_none(results / "table2_minsup_sweep_quantum_full.csv")
    table5 = read_csv_or_none(results / "table5_final.csv")
    selected = read_csv_or_none(results / "table5_selected_minsup.csv")
    candidates = read_csv_or_none(results / "table5_candidates.csv")

    ratio1 = qfm_yu_ratio(table1, "tx")
    ratio2 = qfm_yu_ratio(table2, "minsup")
    t5pivot = table5_pivot(table5)
    t5status = status_summary(table5)
    cand_status = status_summary(candidates)
    cand_compact = compact_candidates(candidates)

    safe_write_csv(ratio1, out_dir / "table1_qfm_yu_ratio.csv")
    safe_write_csv(ratio2, out_dir / "table2_qfm_yu_ratio.csv")
    safe_write_csv(t5pivot, out_dir / "table5_runtime_pivot.csv")
    safe_write_csv(t5status, out_dir / "table5_status_summary.csv")
    safe_write_csv(cand_status, out_dir / "table5_candidates_status_summary.csv")
    safe_write_csv(cand_compact, out_dir / "table5_candidates_compact.csv")

    report: List[str] = []
    report.append("# Experiment Results Summary\n")
    report.append(f"Results directory: `{results}`\n")
    report.append("## Input file availability\n")
    for name, df in [
        ("table1_tx_sweep_quantum_full.csv", table1),
        ("table2_minsup_sweep_quantum_full.csv", table2),
        ("table5_final.csv", table5),
        ("table5_selected_minsup.csv", selected),
        ("table5_candidates.csv", candidates),
    ]:
        if df is None:
            report.append(f"- `{name}`: MISSING\n")
        else:
            report.append(f"- `{name}`: {len(df)} rows; datasets={dataset_list(df)}; methods={method_list(df)}\n")

    report.append("\n## Table 1: QFM/Yu tx-sweep runtime ratio\n")
    report.append(df_to_md(ratio1, args.max_report_rows))

    report.append("\n## Table 2: QFM/Yu minsup-sweep runtime ratio\n")
    report.append(df_to_md(ratio2, args.max_report_rows))

    report.append("\n## Table 5: selected minsup\n")
    report.append(df_to_md(selected, args.max_report_rows))

    report.append("\n## Table 5: runtime pivot\n")
    report.append(df_to_md(t5pivot, args.max_report_rows))

    report.append("\n## Table 5: final status summary\n")
    report.append(df_to_md(t5status, args.max_report_rows))

    report.append("\n## Table 5 candidates: status summary\n")
    report.append(df_to_md(cand_status, args.max_report_rows))

    # Quick warnings / checks
    report.append("\n## Quick checks\n")
    if table1 is not None:
        methods = set(method_list(table1))
        report.append(f"- Table 1 has QFM/Yu: `{ {'QFM','Yu'}.issubset(methods) }`\n")
    if table2 is not None:
        methods = set(method_list(table2))
        report.append(f"- Table 2 has QFM/Yu: `{ {'QFM','Yu'}.issubset(methods) }`\n")
    if table5 is not None:
        methods = set(method_list(table5))
        want = {"QFM", "Yu", "FP-Growth", "Eclat", "Hamm", "CICLAD"}
        report.append(f"- Table 5 has expected methods: `{want.issubset(methods)}`; found={sorted(methods)}\n")
    for label, rat in [("Table 1", ratio1), ("Table 2", ratio2)]:
        if rat is not None and "QFM_over_Yu" in rat.columns:
            ok = int((rat["QFM_over_Yu"] < 1).sum())
            total = int(rat["QFM_over_Yu"].notna().sum())
            report.append(f"- {label}: QFM faster than Yu in {ok}/{total} points.\n")
    if t5pivot is not None and "QFM" in t5pivot.columns:
        for col in [c for c in t5pivot.columns if c.startswith("QFM_over_")]:
            ok = int((t5pivot[col] < 1).sum())
            total = int(t5pivot[col].notna().sum())
            report.append(f"- Table 5: {col}<1 in {ok}/{total} selected points.\n")

    report_path = out_dir / "summary_report.md"
    report_path.write_text("".join(report), encoding="utf-8")
    print(f"[ok] wrote {report_path}")

    print("\n=== Quick summary ===")
    print(f"Output directory: {out_dir}")
    for path in [
        out_dir / "summary_report.md",
        out_dir / "table1_qfm_yu_ratio.csv",
        out_dir / "table2_qfm_yu_ratio.csv",
        out_dir / "table5_runtime_pivot.csv",
        out_dir / "table5_status_summary.csv",
        out_dir / "table5_candidates_compact.csv",
    ]:
        if path.exists():
            print(f"  {path}")

    # Print compact important tables to terminal.
    print("\n=== Table 1 QFM/Yu ratio ===")
    print((ratio1.head(args.max_report_rows).to_string(index=False) if ratio1 is not None else "missing"))
    print("\n=== Table 2 QFM/Yu ratio ===")
    print((ratio2.head(args.max_report_rows).to_string(index=False) if ratio2 is not None else "missing"))
    print("\n=== Table 5 runtime pivot ===")
    print((t5pivot.to_string(index=False) if t5pivot is not None else "missing"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
