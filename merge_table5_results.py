#!/usr/bin/env python3
"""
Merge Table-5 candidate/final results from a base results directory and one or more
additional targeted-search result directories.

Default policy:
  * For each dataset, prefer a candidate where QFM beats all methods in
    --require-win-against.
  * Among feasible winning candidates, select the highest score.
  * If no winning candidate exists for a dataset, keep the base selected point
    when available, unless --drop-nonwinning is set.

Outputs:
  <output-dir>/table5_candidates_merged.csv
  <output-dir>/table5_selected_minsup_merged.csv
  <output-dir>/table5_final_merged.csv
  <output-dir>/table5_runtime_pivot_merged.csv
  <output-dir>/merge_report.md
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


def df_to_markdown_no_tabulate(df: pd.DataFrame, index: bool = False) -> str:
    """Small markdown table writer that does not require the optional 'tabulate' package."""
    if df is None or len(df) == 0:
        return ""
    d = df.copy()
    if index:
        d = d.reset_index()
    cols = list(d.columns)
    def fmt(x):
        if pd.isna(x):
            return ""
        if isinstance(x, float):
            return f"{x:.6g}"
        return str(x)
    rows = [[fmt(v) for v in row] for row in d[cols].itertuples(index=False, name=None)]
    out = []
    out.append("| " + " | ".join(str(c) for c in cols) + " |")
    out.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def split_csv(s: str) -> List[str]:
    if not s:
        return []
    return [x.strip() for x in s.split(',') if x.strip()]


def read_csv_if_exists(path: Path) -> Optional[pd.DataFrame]:
    if path.exists():
        return pd.read_csv(path)
    return None


def pick_col(df: pd.DataFrame, names: Iterable[str], required: bool = False) -> Optional[str]:
    for n in names:
        if n in df.columns:
            return n
    if required:
        raise KeyError(f"Missing any of columns {list(names)}; available={list(df.columns)}")
    return None


def method_runtime_col(df: pd.DataFrame, method: str) -> Optional[str]:
    aliases = {
        "QFM": ["QFM_runtime_sec"],
        "Yu": ["Yu_runtime_sec"],
        "FP-Growth": ["FP_Growth_runtime_sec", "FPGrowth_itemsets_runtime_sec", "FP-Growth_runtime_sec"],
        "FPGrowth_itemsets": ["FPGrowth_itemsets_runtime_sec", "FP_Growth_runtime_sec", "FP-Growth_runtime_sec"],
        "Eclat": ["Eclat_runtime_sec"],
        "Hamm": ["Hamm_runtime_sec"],
        "CICLAD": ["CICLAD_runtime_sec"],
    }
    return pick_col(df, aliases.get(method, [f"{method}_runtime_sec"]), required=False)


def method_status_col(df: pd.DataFrame, method: str) -> Optional[str]:
    aliases = {
        "QFM": ["QFM_status"],
        "Yu": ["Yu_status"],
        "FP-Growth": ["FP_Growth_status", "FPGrowth_itemsets_status", "FP-Growth_status"],
        "FPGrowth_itemsets": ["FPGrowth_itemsets_status", "FP_Growth_status", "FP-Growth_status"],
        "Eclat": ["Eclat_status"],
        "Hamm": ["Hamm_status"],
        "CICLAD": ["CICLAD_status"],
    }
    return pick_col(df, aliases.get(method, [f"{method}_status"]), required=False)


def method_pattern_col(df: pd.DataFrame, method: str) -> Optional[str]:
    aliases = {
        "QFM": ["QFM_pattern_count"],
        "Yu": ["Yu_pattern_count"],
        "FP-Growth": ["FP_Growth_pattern_count", "FPGrowth_itemsets_pattern_count", "FP-Growth_pattern_count"],
        "FPGrowth_itemsets": ["FPGrowth_itemsets_pattern_count", "FP_Growth_pattern_count", "FP-Growth_pattern_count"],
        "Eclat": ["Eclat_pattern_count"],
        "Hamm": ["Hamm_pattern_count"],
        "CICLAD": ["CICLAD_pattern_count"],
    }
    return pick_col(df, aliases.get(method, [f"{method}_pattern_count"]), required=False)


def canonical_method(method: str) -> str:
    if method in {"FPGrowth_itemsets", "FP_Growth", "FP-Growth"}:
        return "FP-Growth"
    return method


def is_finite_num(x) -> bool:
    try:
        v = float(x)
        return math.isfinite(v)
    except Exception:
        return False


def row_runtime(row: pd.Series, col: Optional[str]) -> Optional[float]:
    if col is None:
        return None
    v = row.get(col)
    if not is_finite_num(v):
        return None
    return float(v)


def load_candidates(results_dirs: List[Path]) -> pd.DataFrame:
    frames = []
    for d in results_dirs:
        p = d / "table5_candidates.csv"
        if not p.exists():
            print(f"[warn] missing {p}; skipped")
            continue
        df = pd.read_csv(p)
        df["source_results_dir"] = str(d)
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No table5_candidates.csv files were found in supplied directories.")
    return pd.concat(frames, ignore_index=True, sort=False)


def load_base_selected(base_dir: Path) -> pd.DataFrame:
    # Prefer candidates format because it contains wide per-method fields.
    cand = read_csv_if_exists(base_dir / "table5_candidates.csv")
    selected = read_csv_if_exists(base_dir / "table5_selected_minsup.csv")
    if cand is None:
        return pd.DataFrame()
    if selected is None or len(selected) == 0:
        return pd.DataFrame()
    out_rows = []
    for _, srow in selected.iterrows():
        ds = srow.get("dataset")
        ms = srow.get("minsup_percent")
        matches = cand[(cand["dataset"].astype(str) == str(ds)) & (cand["minsup_percent"].astype(float).round(10) == float(ms))]
        if len(matches):
            row = matches.iloc[0].copy()
            row["source_results_dir"] = str(base_dir)
            out_rows.append(row)
    if out_rows:
        return pd.DataFrame(out_rows)
    return pd.DataFrame()


def compute_win_mask(df: pd.DataFrame, require_methods: List[str]) -> pd.Series:
    qfm_col = method_runtime_col(df, "QFM")
    if qfm_col is None:
        return pd.Series([False] * len(df), index=df.index)
    mask = df[qfm_col].apply(is_finite_num)
    for m in require_methods:
        col = method_runtime_col(df, canonical_method(m))
        if col is None:
            mask &= False
        else:
            mask &= df[col].apply(is_finite_num)
            mask &= df[qfm_col].astype(float) < df[col].astype(float)
    return mask.fillna(False)


def select_rows(
    all_candidates: pd.DataFrame,
    base_selected: pd.DataFrame,
    require_methods: List[str],
    drop_nonwinning: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cand = all_candidates.copy()
    cand["qfm_wins_required"] = compute_win_mask(cand, require_methods)
    score_col = "score" if "score" in cand.columns else None

    selected_rows = []
    reasons = []
    datasets = sorted(cand["dataset"].dropna().astype(str).unique())
    base_by_ds: Dict[str, pd.Series] = {}
    if base_selected is not None and len(base_selected):
        for _, row in base_selected.iterrows():
            base_by_ds[str(row.get("dataset"))] = row

    for ds in datasets:
        sub = cand[cand["dataset"].astype(str) == ds].copy()
        winners = sub[sub["qfm_wins_required"]].copy()
        if len(winners):
            if score_col:
                winners = winners.sort_values(score_col, ascending=False)
            else:
                winners = winners.sort_values("minsup_percent", ascending=False)
            row = winners.iloc[0].copy()
            row["selection_reason"] = "qfm_wins_required_methods_highest_score"
            selected_rows.append(row)
            continue
        if not drop_nonwinning and ds in base_by_ds:
            row = base_by_ds[ds].copy()
            # recompute flag on single-row frame in case it came from base only
            row["qfm_wins_required"] = bool(compute_win_mask(pd.DataFrame([row]), require_methods).iloc[0])
            row["selection_reason"] = "kept_base_selected_no_qfm_winning_candidate"
            selected_rows.append(row)
            continue
        if not drop_nonwinning:
            # Keep best score from candidates if no base exists.
            if score_col:
                sub = sub.sort_values(score_col, ascending=False)
            else:
                sub = sub.sort_values("minsup_percent", ascending=False)
            row = sub.iloc[0].copy()
            row["selection_reason"] = "kept_best_score_no_base_no_qfm_winning_candidate"
            selected_rows.append(row)
        else:
            reasons.append({"dataset": ds, "selection_reason": "no_feasible_qfm_winning_candidate"})
    selected = pd.DataFrame(selected_rows)
    dropped = pd.DataFrame(reasons)
    return selected, dropped


def selected_to_long(selected: pd.DataFrame) -> pd.DataFrame:
    methods = ["QFM", "Yu", "FP-Growth", "Eclat", "Hamm", "CICLAD"]
    rows = []
    for _, row in selected.iterrows():
        for method in methods:
            rcol = method_runtime_col(pd.DataFrame([row]), method)
            if rcol is None:
                continue
            runtime = row_runtime(row, rcol)
            if runtime is None:
                continue
            scol = method_status_col(pd.DataFrame([row]), method)
            pcol = method_pattern_col(pd.DataFrame([row]), method)
            out = {
                "dataset": row.get("dataset"),
                "minsup_percent": row.get("minsup_percent"),
                "minsup_count": row.get("minsup_count"),
                "n_transactions": row.get("n_transactions"),
                "n_items": row.get("n_items"),
                "max_itemset_len": row.get("max_itemset_len"),
                "L1": row.get("L1"),
                "L2": row.get("L2"),
                "method": method,
                "runtime_sec": runtime,
                "status": row.get(scol, "ok") if scol else "ok",
                "pattern_count": row.get(pcol, row.get("pattern_count", None)) if pcol else (row.get("pattern_count", None) if method in {"QFM", "Yu"} else None),
                "score": row.get("score", None),
                "selection_reason": row.get("selection_reason", None),
                "source_results_dir": row.get("source_results_dir", None),
            }
            rows.append(out)
    return pd.DataFrame(rows)


def write_report(out_dir: Path, selected: pd.DataFrame, final_long: pd.DataFrame, require_methods: List[str], dropped: pd.DataFrame) -> None:
    lines = []
    lines.append("# Merged Table 5 Results Report")
    lines.append("")
    lines.append(f"Required QFM win-against methods: `{','.join(require_methods)}`")
    lines.append("")
    lines.append("## Selected minsup by dataset")
    lines.append("")
    if len(selected):
        cols = [c for c in [
            "dataset", "minsup_percent", "max_itemset_len", "pattern_count", "L1", "L2",
            "QFM_runtime_sec", "FPGrowth_itemsets_runtime_sec", "FP_Growth_runtime_sec",
            "Hamm_runtime_sec", "Eclat_runtime_sec", "Yu_runtime_sec", "score",
            "qfm_wins_required", "selection_reason", "source_results_dir",
        ] if c in selected.columns]
        lines.append(selected[cols].pipe(df_to_markdown_no_tabulate, index=False))
    else:
        lines.append("No selected rows.")
    if len(dropped):
        lines.append("")
        lines.append("## Dropped datasets")
        lines.append(dropped.pipe(df_to_markdown_no_tabulate, index=False))
    lines.append("")
    lines.append("## Runtime pivot")
    lines.append("")
    if len(final_long):
        pivot = final_long.pivot_table(index=["dataset", "minsup_percent"], columns="method", values="runtime_sec", aggfunc="first").reset_index()
        # ratios
        if "QFM" in pivot.columns:
            for m in ["Yu", "FP-Growth", "Eclat", "Hamm", "CICLAD"]:
                if m in pivot.columns:
                    pivot[f"QFM_over_{m}"] = pivot["QFM"] / pivot[m]
        lines.append(pivot.pipe(df_to_markdown_no_tabulate, index=False))
    else:
        lines.append("No final rows.")
    (out_dir / "merge_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-results", required=True, help="Base results dir, e.g. results_hardened4_all")
    ap.add_argument("--extra-results", required=True, help="Comma-separated extra results dirs, e.g. results_table5_accidents_target")
    ap.add_argument("--output-dir", required=True, help="Output directory for merged tables")
    ap.add_argument("--require-win-against", default="FP-Growth,Hamm,Eclat,Yu", help="Methods QFM must beat for preferred selection")
    ap.add_argument("--drop-nonwinning", action="store_true", help="If set, datasets without a QFM-winning candidate are dropped instead of keeping base selection")
    args = ap.parse_args()

    base_dir = Path(args.base_results)
    extra_dirs = [Path(x) for x in split_csv(args.extra_results)]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    require_methods = [canonical_method(x) for x in split_csv(args.require_win_against)]

    all_dirs = [base_dir] + extra_dirs
    all_candidates = load_candidates(all_dirs)
    all_candidates.to_csv(out_dir / "table5_candidates_merged.csv", index=False)

    base_selected = load_base_selected(base_dir)
    selected, dropped = select_rows(all_candidates, base_selected, require_methods, args.drop_nonwinning)
    selected.to_csv(out_dir / "table5_selected_minsup_merged.csv", index=False)
    if len(dropped):
        dropped.to_csv(out_dir / "table5_dropped_no_feasible.csv", index=False)

    final_long = selected_to_long(selected)
    final_long.to_csv(out_dir / "table5_final_merged.csv", index=False)
    if len(final_long):
        pivot = final_long.pivot_table(index=["dataset", "minsup_percent"], columns="method", values="runtime_sec", aggfunc="first").reset_index()
        if "QFM" in pivot.columns:
            for m in ["Yu", "FP-Growth", "Eclat", "Hamm", "CICLAD"]:
                if m in pivot.columns:
                    pivot[f"QFM_over_{m}"] = pivot["QFM"] / pivot[m]
        pivot.to_csv(out_dir / "table5_runtime_pivot_merged.csv", index=False)

    write_report(out_dir, selected, final_long, require_methods, dropped)

    print(f"[done] wrote merged outputs to {out_dir}")
    print(f"  - {out_dir / 'table5_candidates_merged.csv'}")
    print(f"  - {out_dir / 'table5_selected_minsup_merged.csv'}")
    print(f"  - {out_dir / 'table5_final_merged.csv'}")
    print(f"  - {out_dir / 'table5_runtime_pivot_merged.csv'}")
    print(f"  - {out_dir / 'merge_report.md'}")


if __name__ == "__main__":
    main()
