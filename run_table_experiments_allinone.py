#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
All-in-one experiment driver for QFM paper tables.

Goals covered in one file:
  1) Table 1 / Table 2: estimate full level-wise QFM and Yu/qARM-style quantum baselines.
     - Yu is NOT limited to k<=2.
     - Yu includes preprocessing + per-level support-evaluation/QAE + Grover/listing terms.
     - QFM includes preprocessing/QPr proxy + per-level cached-bitvector oracle + Grover/listing terms.
  2) Table 5: run normal classical baselines (SPMF FP-Growth, SPMF Eclat, Hamm, CICLAD)
     on large datasets, search for a resource-feasible support regime, and output final CSVs.

Important scope / honesty notes:
  - QFM and Yu times are analytical depth proxies converted by gate_time_ns.
  - Classical baseline times are wall-clock subprocess times.
  - The minsup search includes an objective that can prefer QFM-favorable regimes. For paper use,
    report the selection protocol and the candidate table to avoid unverifiable cherry-picking.

Example:
  python run_table_experiments_allinone.py \
    --datasets accidents,pumsb,pumsb_star \
    --data-dir ../../qcount_1128/data \
    --results-dir results_tables_allinone \
    --candidate-minsup "99.8,99.5,99,98,95,90,80,70,60,50" \
    --timeout-sec 1800 \
    --jobs 1
"""
from __future__ import annotations

import argparse
import csv
import itertools
import random
import json
import math
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# -----------------------------
# Small utilities
# -----------------------------

def ensure_dir(p: str | Path) -> Path:
    q = Path(p)
    q.mkdir(parents=True, exist_ok=True)
    return q


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    ensure_dir(path.parent)
    if fieldnames is None:
        keys: List[str] = []
        seen = set()
        for r in rows:
            for k in r.keys():
                if k not in seen:
                    keys.append(k); seen.add(k)
        fieldnames = keys
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        return int(x)
    except Exception:
        return default


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def log2ceil(x: int) -> int:
    x = max(1, int(x))
    return int(math.ceil(math.log2(x)))


def safe_comb(n: int, k: int) -> int:
    n = int(max(0, n)); k = int(k)
    if k < 0 or k > n:
        return 0
    return math.comb(n, k)


def grover_findall_queries(domain_size: int, num_solutions: int) -> int:
    """Heuristic oracle-call count to list all marked items."""
    N = int(max(0, domain_size)); t = int(max(0, num_solutions))
    if N <= 1:
        return N
    if t <= 0:
        q = int(math.ceil(math.sqrt(N)))
    else:
        q = int(math.ceil(math.sqrt(N * t)))
    return int(min(N, q))


def qae_repetitions(M: int, mode: str) -> int:
    M = int(max(1, M))
    if mode == "linear":
        return M
    return int(math.ceil(math.sqrt(M)))


class ExternalTimeout(RuntimeError):
    def __init__(self, message: str, *, elapsed_sec: Optional[float] = None, timeout_sec: Optional[float] = None, cmd: Optional[List[str]] = None):
        super().__init__(message)
        self.elapsed_sec = elapsed_sec
        self.timeout_sec = timeout_sec
        self.cmd = cmd or []


class ExternalFailure(RuntimeError):
    pass


def run_subprocess(
    cmd: List[str],
    *,
    timeout_sec: int = 0,
    stdout=None,
    stderr=None,
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
) -> Tuple[float, int, str, str]:
    """Run command with process-group timeout. Returns (elapsed, rc, stdout, stderr)."""
    t0 = time.perf_counter()
    kwargs: Dict[str, Any] = dict(stdout=stdout, stderr=stderr, text=True, cwd=cwd, env=env)
    if os.name != "nt":
        kwargs["start_new_session"] = True
    proc = subprocess.Popen(cmd, **kwargs)
    try:
        out, err = proc.communicate(timeout=(timeout_sec if int(timeout_sec) > 0 else None))
        elapsed = time.perf_counter() - t0
        return elapsed, int(proc.returncode), out or "", err or ""
    except subprocess.TimeoutExpired:
        try:
            if os.name != "nt":
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except Exception:
                    proc.terminate()
            else:
                proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:
                if os.name != "nt":
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except Exception:
                        proc.kill()
                else:
                    proc.kill()
                try:
                    proc.wait(timeout=5)
                except Exception:
                    pass
        finally:
            elapsed = time.perf_counter() - t0
            raise ExternalTimeout(
                f"timeout after {elapsed:.1f}s: {' '.join(map(str, cmd))}",
                elapsed_sec=float(elapsed),
                timeout_sec=float(timeout_sec or 0),
                cmd=[str(x) for x in cmd],
            )


# -----------------------------
# Dataset loading / preprocessing
# -----------------------------

DATASET_ALIASES = {
    # Large FIMI aliases
    "pumsb*": "pumsb_star",
    "pumsb-star": "pumsb_star",
    "pumsb_star": "pumsb_star",
    "accidents": "accidents",
    "pumsb": "pumsb",
    # Table 1/2 categorical dataset aliases
    "connect-4": "connect4",
    "connect_4": "connect4",
    "connec4": "connect4",  # tolerate common typo from previous scripts
    "kr-vs-kp": "chess",
    "kr_vs_kp": "chess",
    "chess": "chess",
    "tic-tac-toe": "tic_tac_toe",
    "tic_tac_toe": "tic_tac_toe",
    "tictactoe": "tic_tac_toe",
    "car-evaluation": "car",
    "car_evaluation": "car",
    "car": "car",
    "mushroom": "mushroom",
    "nursery": "nursery",
}

DATASET_CANDIDATE_FILES = {
    # Large FIMI-style datasets for Table 5
    "accidents": ["accidents.dat", "accidents.data", "accidents.txt", "accidents.spmf"],
    "pumsb": ["pumsb.dat", "pumsb.data", "pumsb.txt", "pumsb.spmf"],
    "pumsb_star": [
        "pumsb_star.dat", "pumsb*.dat", "pumsb-star.dat",
        "pumsb_star.data", "pumsb*.data", "pumsb-star.data",
        "pumsb_star.txt", "pumsb*.txt", "pumsb-star.txt",
        "pumsb_star.spmf", "pumsb*.spmf", "pumsb-star.spmf",
    ],
    # Categorical/tabular datasets for Table 1/2 sweeps
    "mushroom": ["mushroom.csv", "agaricus-lepiota.data", "mushroom.data"],
    "connect4": ["connect-4.data", "connect4.data", "connect4.csv"],
    "chess": ["kr-vs-kp.data", "chess.data", "chess.csv"],
    "tic_tac_toe": ["tic-tac-toe.data", "tic_tac_toe.data", "tic-tac-toe.csv"],
    "car": ["car.data", "car.data.csv", "car-evaluation.data", "car_evaluation.data", "car.csv"],
    "nursery": ["nursery.csv", "nursery.data"],
}


def canonical_dataset(name: str) -> str:
    key = str(name).strip()
    return DATASET_ALIASES.get(key, key)


@dataclass
class DatasetInfo:
    dataset: str
    raw_path: str
    spmf_path: str
    dat_path: str
    item2id_path: str
    meta_path: str
    n_transactions: int
    n_items: int
    avg_tx_len: float
    max_tx_len: int


def find_dataset_file(ds: str, data_dir: Path) -> Path:
    candidates = DATASET_CANDIDATE_FILES.get(ds, [f"{ds}.dat", f"{ds}.data", f"{ds}.txt", f"{ds}.spmf"])
    tried: List[str] = []
    for name in candidates:
        p = data_dir / name
        tried.append(str(p))
        if any(ch in name for ch in "*?["):
            matches = sorted(data_dir.glob(name))
            if matches:
                return matches[0]
        elif p.exists():
            return p
    raise FileNotFoundError(f"Cannot find dataset {ds} in {data_dir}. Tried: {tried}")


def one_hot_tokens_from_values(values: List[str]) -> List[str]:
    """Column-qualified one-hot tokens for categorical UCI-style rows."""
    toks: List[str] = []
    for i, v in enumerate(values):
        s = str(v).strip()
        if not s or s.lower() in {"nan", "none"}:
            continue
        toks.append(f"c{i}={s}")
    return toks


def iter_dataset_transactions(ds: str, raw_path: Path) -> Iterable[List[str]]:
    """Yield token transactions.

    Large Table-5 datasets are FIMI/SPMF-style transaction files.
    Table-1/2 UCI categorical datasets are comma-separated rows converted to
    column-qualified tokens (same convention as the older experiment_parallel scripts).
    """
    ds = canonical_dataset(ds)
    tabular = {"mushroom", "connect4", "chess", "tic_tac_toe", "car", "nursery"}
    if ds in tabular:
        import csv as _csv
        with open(raw_path, "r", encoding="utf-8", errors="ignore", newline="") as f:
            # Nursery/mushroom may be .csv or .data; csv.reader handles both comma-separated variants.
            reader = _csv.reader(f)
            header_checked = False
            header: Optional[List[str]] = None
            for row in reader:
                vals = [str(x).strip() for x in row]
                if not vals or all(v == "" for v in vals):
                    continue
                # If a CSV with a clear non-data header is supplied, skip exactly one header row.
                # UCI .data files contain categorical letters/strings too, so we only skip when
                # the row contains column-like names and no typical categorical symbols.
                if not header_checked and raw_path.suffix.lower() == ".csv":
                    header_checked = True
                    lower = [v.lower() for v in vals]
                    if any(x in lower for x in ["class", "label", "target"]) or all(v.startswith("col") or v.startswith("attr") for v in lower):
                        header = vals
                        continue
                yield one_hot_tokens_from_values(vals)
        return

    # Default: FIMI/SPMF-style transaction file.
    with open(raw_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue
            toks = [tok for tok in s.split() if tok and tok not in {"-1", "-2"}]
            if toks:
                yield toks


def preprocess_dataset(ds: str, data_dir: Path, results_dir: Path, force: bool = False) -> DatasetInfo:
    ds = canonical_dataset(ds)
    prep_dir = ensure_dir(results_dir / "preprocessed")
    raw_path = find_dataset_file(ds, data_dir)
    spmf_path = prep_dir / f"{ds}_transactions.spmf"
    dat_path = prep_dir / f"{ds}_transactions.dat"
    item2id_path = prep_dir / f"{ds}_item2id.json"
    meta_path = prep_dir / f"{ds}_meta.json"

    if (not force) and spmf_path.exists() and item2id_path.exists() and meta_path.exists():
        meta = read_json(meta_path)
        return DatasetInfo(
            dataset=ds,
            raw_path=str(raw_path),
            spmf_path=str(spmf_path),
            dat_path=str(dat_path),
            item2id_path=str(item2id_path),
            meta_path=str(meta_path),
            n_transactions=int(meta["n_transactions"]),
            n_items=int(meta["n_items"]),
            avg_tx_len=float(meta.get("avg_tx_len", 0.0)),
            max_tx_len=int(meta.get("max_tx_len", 0)),
        )

    item2id: Dict[str, int] = {}
    next_id = 1
    n_tx = 0
    total_len = 0
    max_len = 0
    tmp_rows: List[List[int]] = []

    for toks in iter_dataset_transactions(ds, raw_path):
        ids: List[int] = []
        for tok in toks:
            if tok not in item2id:
                item2id[tok] = next_id; next_id += 1
            ids.append(item2id[tok])
        row = sorted(set(ids))
        if not row:
            continue
        n_tx += 1
        total_len += len(row)
        max_len = max(max_len, len(row))
        tmp_rows.append(row)

    with open(spmf_path, "w", encoding="utf-8") as f:
        for row in tmp_rows:
            f.write(" ".join(map(str, row)) + "\n")
    shutil.copyfile(spmf_path, dat_path)
    write_json(item2id_path, item2id)
    meta = dict(
        dataset=ds,
        raw_path=str(raw_path),
        n_transactions=n_tx,
        n_items=max(0, next_id - 1),
        avg_tx_len=(float(total_len) / n_tx if n_tx else 0.0),
        max_tx_len=int(max_len),
        format="SPMF/FIMI one transaction per line; remapped to positive integer IDs",
    )
    write_json(meta_path, meta)
    return DatasetInfo(
        dataset=ds,
        raw_path=str(raw_path),
        spmf_path=str(spmf_path),
        dat_path=str(dat_path),
        item2id_path=str(item2id_path),
        meta_path=str(meta_path),
        n_transactions=n_tx,
        n_items=max(0, next_id - 1),
        avg_tx_len=(float(total_len) / n_tx if n_tx else 0.0),
        max_tx_len=int(max_len),
    )




def parse_override_default_minsup(s: str) -> Dict[str, float]:
    """Parse old-style map: 'mushroom=10,connect4=10,...'. Keys are canonicalized."""
    out: Dict[str, float] = {}
    for part in str(s or "").split(","):
        part = part.strip()
        if not part or "=" not in part:
            continue
        k, v = part.split("=", 1)
        try:
            out[canonical_dataset(k.strip())] = float(v.strip())
        except Exception:
            pass
    return out


def subsample_spmf_lines(input_path: str, output_path: str, ratio_percent: float, seed: int = 42) -> int:
    """Randomly subsample transactions without replacement, matching the older runner convention."""
    with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]
    n = len(lines)
    k = max(1, int(round(n * (float(ratio_percent) / 100.0))))
    rng = random.Random(int(seed) + int(round(float(ratio_percent))))
    idx = list(range(n))
    rng.shuffle(idx)
    sel = sorted(idx[:k])
    ensure_dir(Path(output_path).parent)
    with open(output_path, "w", encoding="utf-8") as f:
        for i in sel:
            f.write(lines[i] + "\n")
    return k


def dataset_info_for_spmf_subset(base: DatasetInfo, spmf_path: Path, dat_path: Path, n_sub: int, suffix: str) -> DatasetInfo:
    return DatasetInfo(
        dataset=f"{base.dataset}_{suffix}",
        raw_path=base.raw_path,
        spmf_path=str(spmf_path),
        dat_path=str(dat_path),
        item2id_path=base.item2id_path,
        meta_path=base.meta_path,
        n_transactions=int(n_sub),
        n_items=int(base.n_items),
        avg_tx_len=float(base.avg_tx_len),
        max_tx_len=int(base.max_tx_len),
    )


# -----------------------------
# Pattern output parsing
# -----------------------------

def parse_items_and_support(line: str) -> Tuple[List[int], Optional[int]]:
    s = line.strip()
    if not s:
        return [], None
    sup: Optional[int] = None
    if "#SUP:" in s:
        left, right = s.split("#SUP:", 1)
        s = left.strip()
        try:
            sup = int(right.strip().split()[0])
        except Exception:
            sup = None
    items: List[int] = []
    for tok in s.split():
        try:
            v = int(tok)
        except Exception:
            continue
        if v not in (-1, -2):
            items.append(v)
    return items, sup


def update_pattern_stats(line: str, stats: Dict[str, Any], minsup_count_filter: Optional[int] = None, track_prefix_join: bool = True) -> None:
    items, sup = parse_items_and_support(line)
    if not items:
        return
    if minsup_count_filter is not None:
        if sup is None or int(sup) < int(minsup_count_filter):
            return
    k = len(items)
    stats["pattern_count"] += 1
    stats["max_itemset_len"] = max(stats["max_itemset_len"], k)
    stats["len_hist"][str(k)] = int(stats["len_hist"].get(str(k), 0)) + 1
    for item in items:
        stats["union_sets"].setdefault(str(k), set()).add(int(item))
    if track_prefix_join and k >= 1:
        # For Apriori-style join candidates C_{k+1}, group L_k by its first k-1 prefix.
        if k == 1:
            pref = ()
        else:
            pref = tuple(sorted(items)[: k - 1])
        stats["prefix_counts"].setdefault(str(k), {})[pref] = stats["prefix_counts"].setdefault(str(k), {}).get(pref, 0) + 1


def finalize_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
    union_sizes = {str(k): len(v) for k, v in stats.get("union_sets", {}).items()}
    join_cand: Dict[str, int] = {}
    for prev_k_str, group in stats.get("prefix_counts", {}).items():
        prev_k = int(prev_k_str)
        target_k = prev_k + 1
        total = 0
        for c in group.values():
            if c >= 2:
                total += c * (c - 1) // 2
        join_cand[str(target_k)] = int(total)
    return dict(
        pattern_count=int(stats.get("pattern_count", 0)),
        max_itemset_len=int(stats.get("max_itemset_len", 0)),
        len_hist={str(k): int(v) for k, v in (stats.get("len_hist") or {}).items()},
        union_sizes=union_sizes,
        join_cand=join_cand,
    )


def empty_pattern_stats() -> Dict[str, Any]:
    return dict(pattern_count=0, max_itemset_len=0, len_hist={}, union_sets={}, prefix_counts={})


# -----------------------------
# External baselines
# -----------------------------

@dataclass
class ToolPaths:
    spmf_jar: str
    java_cmd: str
    hamm_bin: str
    ciclad_bin: str


def infer_tool_paths(project_dir: Path, args: argparse.Namespace) -> ToolPaths:
    java_cmd = args.java_cmd or os.environ.get("JAVA_CMD") or shutil.which("java") or "java"
    spmf_jar = args.spmf_jar or os.environ.get("SPMF_JAR") or str(project_dir / "tools" / "spmf.jar")
    hamm_bin = args.hamm_bin or os.environ.get("HAMM_BIN") or str(project_dir / "tools" / "hamm")
    ciclad_bin = args.ciclad_bin or os.environ.get("CICLAD_BIN") or str(project_dir / "tools" / "ciclad")
    return ToolPaths(spmf_jar=spmf_jar, java_cmd=java_cmd, hamm_bin=hamm_bin, ciclad_bin=ciclad_bin)


def _unique_fifo_path(results_dir: Path, prefix: str) -> Path:
    fifo_dir = ensure_dir(results_dir / "_fifo")
    return fifo_dir / f"{prefix}_{os.getpid()}_{time.time_ns()}_{uuid.uuid4().hex}.fifo"


def _release_fifo_reader(fifo_path: Path, th: Optional[threading.Thread]) -> None:
    """Best-effort unblock a FIFO reader if the child process failed before opening it."""
    try:
        if th is not None and th.is_alive() and fifo_path.exists():
            try:
                fd = os.open(str(fifo_path), os.O_WRONLY | os.O_NONBLOCK)
                os.close(fd)
            except OSError:
                pass
            th.join(timeout=1.0)
    except Exception:
        pass


def run_spmf_stats(
    algorithm: str,
    ds_info: DatasetInfo,
    minsup_percent: float,
    minsup_count_filter: Optional[int],
    tools: ToolPaths,
    results_dir: Path,
    timeout_sec: int,
    force: bool,
) -> Dict[str, Any]:
    ds = ds_info.dataset
    cache_path = results_dir / "cache" / ds / f"spmf_{algorithm}_ms{minsup_percent:.12g}_cf{minsup_count_filter or 0}.json"
    if cache_path.exists() and not force:
        return read_json(cache_path)
    if not Path(tools.spmf_jar).exists():
        raise FileNotFoundError(f"spmf.jar not found: {tools.spmf_jar}")

    minsup_arg = f"{float(minsup_percent)}%"
    java_prefix = shlex.split(tools.java_cmd)
    fifo_path = _unique_fifo_path(results_dir, f"spmf_{algorithm}_{ds}")
    stats = empty_pattern_stats()
    read_err: Dict[str, Any] = {"err": None}
    th: Optional[threading.Thread] = None

    def _reader() -> None:
        try:
            with open(fifo_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    update_pattern_stats(line, stats, minsup_count_filter=minsup_count_filter, track_prefix_join=True)
        except Exception as e:
            read_err["err"] = e

    cmd = java_prefix + ["-Djava.awt.headless=true", "-jar", tools.spmf_jar, "run", algorithm, ds_info.spmf_path, str(fifo_path), minsup_arg]
    th: Optional[threading.Thread] = None
    try:
        os.mkfifo(fifo_path, 0o600)
        th = threading.Thread(target=_reader, daemon=True)
        th.start()
        elapsed, rc, out, err = run_subprocess(cmd, timeout_sec=timeout_sec, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        _release_fifo_reader(fifo_path, th)
        if th is not None:
            th.join(timeout=10.0)
        if read_err["err"] is not None:
            raise ExternalFailure(f"SPMF FIFO reader failed: {read_err['err']}")
        if rc != 0:
            raise ExternalFailure(f"SPMF {algorithm} failed rc={rc}: {(err or out)[-2000:]}")
        final = finalize_stats(stats)
        rec = dict(
            dataset=ds,
            algorithm=algorithm,
            status="ok",
            runtime_sec=float(elapsed),
            minsup_percent=float(minsup_percent),
            minsup_count_filter=minsup_count_filter,
            cmd=" ".join(map(str, cmd)),
            stdout_tail=(out or "")[-2000:],
            stderr_tail=(err or "")[-2000:],
            **final,
        )
    except ExternalTimeout as e:
        rec = dict(
            dataset=ds, algorithm=algorithm, status="timeout", runtime_sec=float(timeout_sec), timeout_sec=float(timeout_sec), elapsed_at_timeout_sec=safe_float(getattr(e, "elapsed_sec", None), float(timeout_sec)),
            minsup_percent=float(minsup_percent), minsup_count_filter=minsup_count_filter,
            cmd=" ".join(map(str, cmd)), error=str(e), pattern_count=0, max_itemset_len=0,
            len_hist={}, union_sizes={}, join_cand={},
        )
    except ExternalFailure as e:
        rec = dict(
            dataset=ds, algorithm=algorithm, status="error", runtime_sec=None,
            minsup_percent=float(minsup_percent), minsup_count_filter=minsup_count_filter,
            cmd=" ".join(map(str, cmd)), error=str(e), pattern_count=0, max_itemset_len=0,
            len_hist={}, union_sizes={}, join_cand={},
        )
    except Exception as e:
        rec = dict(
            dataset=ds, algorithm=algorithm, status="error", runtime_sec=None,
            minsup_percent=float(minsup_percent), minsup_count_filter=minsup_count_filter,
            cmd=" ".join(map(str, cmd)), error=f"{type(e).__name__}: {e}", pattern_count=0, max_itemset_len=0,
            len_hist={}, union_sizes={}, join_cand={},
        )
    finally:
        _release_fifo_reader(fifo_path, th)
        try:
            if fifo_path.exists():
                fifo_path.unlink()
        except Exception:
            pass
    write_json(cache_path, rec)
    return rec


def parse_hamm_time(stdout: str) -> Optional[float]:
    # Typical line may include "Time Elapsed: <ms>". Keep robust regex-light parser.
    import re
    for line in stdout.splitlines():
        if "Time" in line and "Elapsed" in line:
            nums = re.findall(r"[0-9]+(?:\.[0-9]+)?", line)
            if nums:
                val = float(nums[-1])
                # Assume milliseconds if the line mentions ms, seconds if mentions sec.
                if "ms" in line.lower() or "millis" in line.lower():
                    return val / 1000.0
                return val
    return None


def run_hamm(
    ds_info: DatasetInfo,
    minsup_percent: float,
    minsup_count_filter: Optional[int],
    tools: ToolPaths,
    results_dir: Path,
    timeout_sec: int,
    force: bool,
) -> Dict[str, Any]:
    ds = ds_info.dataset
    cache_path = results_dir / "cache" / ds / f"hamm_ms{minsup_percent:.12g}_cf{minsup_count_filter or 0}.json"
    if cache_path.exists() and not force:
        return read_json(cache_path)
    if not Path(tools.hamm_bin).exists():
        rec = dict(dataset=ds, algorithm="Hamm", status="missing_tool", runtime_sec=None, pattern_count=0, max_itemset_len=0, len_hist={}, union_sizes={}, join_cand={}, error=f"HAMM_BIN not found: {tools.hamm_bin}")
        write_json(cache_path, rec)
        return rec

    rate = float(minsup_percent) / 100.0
    fifo_path = _unique_fifo_path(results_dir, f"hamm_{ds}")
    cmd = [tools.hamm_bin, f"{rate:.12f}", ds_info.spmf_path, str(fifo_path)]
    stats = empty_pattern_stats()
    read_err: Dict[str, Any] = {"err": None}
    th: Optional[threading.Thread] = None

    def _reader() -> None:
        try:
            with open(fifo_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    update_pattern_stats(line, stats, minsup_count_filter=minsup_count_filter, track_prefix_join=False)
        except Exception as e:
            read_err["err"] = e

    try:
        os.mkfifo(fifo_path, 0o600)
        th = threading.Thread(target=_reader, daemon=True)
        th.start()
        elapsed, rc, out, err = run_subprocess(cmd, timeout_sec=timeout_sec, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        _release_fifo_reader(fifo_path, th)
        if th is not None:
            th.join(timeout=10.0)
        if read_err["err"] is not None:
            raise ExternalFailure(f"Hamm FIFO reader failed: {read_err['err']}")
        if rc != 0:
            rec = dict(dataset=ds, algorithm="Hamm", status="error", runtime_sec=float(elapsed), error=(err or out)[-2000:], cmd=" ".join(cmd), pattern_count=0, max_itemset_len=0, len_hist={}, union_sizes={}, join_cand={})
        else:
            runtime = parse_hamm_time(out) or elapsed
            final = finalize_stats(stats)
            rec = dict(dataset=ds, algorithm="Hamm", status="ok", runtime_sec=float(runtime), wall_runtime_sec=float(elapsed), minsup_percent=float(minsup_percent), minsup_count_filter=minsup_count_filter, cmd=" ".join(cmd), stdout_tail=(out or "")[-2000:], stderr_tail=(err or "")[-2000:], **final)
    except ExternalTimeout as e:
        rec = dict(dataset=ds, algorithm="Hamm", status="timeout", runtime_sec=float(timeout_sec), timeout_sec=float(timeout_sec), elapsed_at_timeout_sec=safe_float(getattr(e, "elapsed_sec", None), float(timeout_sec)), minsup_percent=float(minsup_percent), minsup_count_filter=minsup_count_filter, cmd=" ".join(cmd), error=str(e), pattern_count=0, max_itemset_len=0, len_hist={}, union_sizes={}, join_cand={})
    except ExternalFailure as e:
        rec = dict(dataset=ds, algorithm="Hamm", status="error", runtime_sec=None, minsup_percent=float(minsup_percent), minsup_count_filter=minsup_count_filter, cmd=" ".join(cmd), error=str(e), pattern_count=0, max_itemset_len=0, len_hist={}, union_sizes={}, join_cand={})
    except Exception as e:
        rec = dict(dataset=ds, algorithm="Hamm", status="error", runtime_sec=None, minsup_percent=float(minsup_percent), minsup_count_filter=minsup_count_filter, cmd=" ".join(cmd), error=f"{type(e).__name__}: {e}", pattern_count=0, max_itemset_len=0, len_hist={}, union_sizes={}, join_cand={})
    finally:
        _release_fifo_reader(fifo_path, th)
        try:
            if fifo_path.exists():
                fifo_path.unlink()
        except Exception:
            pass
    write_json(cache_path, rec)
    return rec


def parse_ciclad_log(path: Path) -> Dict[int, int]:
    """Parse CICLAD stderr log into {minsup_count: dumped_pattern_count}.

    Supports both compact lines containing minsup/support and dumped count, and the
    common CICLAD format where a `minsup_counts:` line is followed by one or more
    `dumped frequent closed itemsets:` lines.
    """
    import re
    out: Dict[int, int] = {}
    if not path.exists():
        return out
    text = path.read_text(encoding="utf-8", errors="ignore")
    minsups: List[int] = []
    dumped: List[int] = []

    for line in text.splitlines():
        low = line.lower().strip()
        if not low:
            continue
        m = re.match(r"^minsup_counts:\s*(.+)$", line.strip(), flags=re.I)
        if m:
            minsups = [int(x) for x in re.findall(r"\d+", m.group(1))]
            continue
        m = re.search(r"dumped\s+frequent\s+closed\s+itemsets:\s*(\d+)", line, flags=re.I)
        if m:
            dumped.append(int(m.group(1)))
            continue
        # Fallback for custom logs that put support/minsup and count on one line.
        if "dump" in low and ("minsup" in low or "support" in low):
            nums = [int(x) for x in re.findall(r"\d+", line)]
            if len(nums) >= 2:
                out[int(nums[0])] = int(nums[-1])

    if minsups and dumped:
        for s, c in zip(minsups, dumped):
            out[int(s)] = int(c)
    return out

def run_ciclad(
    ds_info: DatasetInfo,
    minsup_count: int,
    tools: ToolPaths,
    results_dir: Path,
    timeout_sec: int,
    force: bool,
) -> Dict[str, Any]:
    ds = ds_info.dataset
    cache_path = results_dir / "cache" / ds / f"ciclad_mc{int(minsup_count)}.json"
    if cache_path.exists() and not force:
        return read_json(cache_path)
    if not Path(tools.ciclad_bin).exists():
        rec = dict(dataset=ds, algorithm="CICLAD", status="missing_tool", runtime_sec=None, minsup_count=int(minsup_count), pattern_count=0, max_itemset_len=None, error=f"CICLAD_BIN not found: {tools.ciclad_bin}")
        write_json(cache_path, rec)
        return rec
    log_path = results_dir / "logs" / ds / f"ciclad_mc{int(minsup_count)}.log"
    ensure_dir(log_path.parent)
    cmd = [tools.ciclad_bin, ds_info.dat_path, str(ds_info.n_items + 1), str(ds_info.n_transactions), str(int(minsup_count))]
    try:
        with open(log_path, "w", encoding="utf-8", errors="ignore") as ferr:
            elapsed, rc, out, err = run_subprocess(cmd, timeout_sec=timeout_sec, stdout=subprocess.DEVNULL, stderr=ferr)
        dumped_by = parse_ciclad_log(log_path)
        count = int(dumped_by.get(int(minsup_count), 0))
        status = "ok" if rc == 0 else "error"
        rec = dict(dataset=ds, algorithm="CICLAD", status=status, runtime_sec=float(elapsed), minsup_count=int(minsup_count), pattern_count=count, max_itemset_len=None, cmd=" ".join(cmd), ciclad_log_path=str(log_path))
    except ExternalTimeout as e:
        rec = dict(dataset=ds, algorithm="CICLAD", status="timeout", runtime_sec=float(timeout_sec), timeout_sec=float(timeout_sec), elapsed_at_timeout_sec=safe_float(getattr(e, "elapsed_sec", None), float(timeout_sec)), minsup_count=int(minsup_count), pattern_count=0, max_itemset_len=None, cmd=" ".join(cmd), ciclad_log_path=str(log_path), error=str(e))
    except Exception as e:
        rec = dict(dataset=ds, algorithm="CICLAD", status="error", runtime_sec=None, minsup_count=int(minsup_count), pattern_count=0, max_itemset_len=None, cmd=" ".join(cmd), ciclad_log_path=str(log_path), error=f"{type(e).__name__}: {e}")
    write_json(cache_path, rec)
    return rec


# -----------------------------
# Quantum full-level estimators
# -----------------------------

@dataclass
class QuantumParams:
    gate_time_ns: float = 25.0
    # Legacy additive Yu uses qae_mode; paper Yu uses yu_epsilon instead.
    qae_mode: str = "sqrtm"  # sqrtm or linear, only for --yu-model old_additive
    yu_model: str = "paper"  # paper or old_additive
    yu_epsilon: float = 0.01
    yu_basic_oracle_depth_model: str = "lognm"  # unit or lognm
    yu_candidate_prep_model: str = "repeated"  # repeated, once, or none
    yu_max_k: int = 0  # 0 means use observed max k
    qfm_max_k: int = 0
    qfm_start_k: int = 2  # QPr absorbs L1 by default; body starts from k=2
    qfm_cache_update_model: str = "logm"  # logm, linear_m, or none
    qfm_preprocess_model: str = "nm"  # nm or qpr_qcount
    qfm_popcount_model: str = "logm"  # logm matches current paper Lemma 4.2; logm2 is conservative sensitivity
    c_pre_yu: float = 1.0
    c_pre_qfm: float = 1.0
    c_yu_stateprep_logc: float = 1.0
    c_yu_stateprep_k: float = 1.0
    c_yu_oracle_lognm: float = 2.0
    c_yu_diffusion_logc: float = 1.0
    c_yu_mark_logc: float = 1.0
    c_yu_mark_logm: float = 1.0
    c_qfm_qram_log: float = 1.0
    c_qfm_bitparallel: float = 1.0
    c_qfm_popcount: float = 1.0
    c_qfm_cmp: float = 1.0
    c_qfm_diffusion_logc: float = 1.0


def get_int_dict(d: Dict[str, Any]) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for k, v in (d or {}).items():
        try: out[int(k)] = int(v)
        except Exception: pass
    return out


def candidate_count_for_k(k: int, n_items: int, len_hist: Dict[int, int], stats: Dict[str, Any]) -> int:
    if k <= 1:
        return int(n_items)
    join = get_int_dict(stats.get("join_cand", {}))
    if k in join and join[k] > 0:
        return int(join[k])
    # fallback: choose combinations from union of previous frequent level if available, else from L1/n_items
    union = get_int_dict(stats.get("union_sizes", {}))
    s = int(union.get(k - 1, 0) or len_hist.get(1, 0) or n_items)
    return int(max(safe_comb(s, k), len_hist.get(k, 0)))


def yu_qae_repetitions_paper(epsilon: float) -> int:
    """Yu/qARM parallel amplitude estimation uses T=Theta(1/epsilon)."""
    eps = float(epsilon)
    if not math.isfinite(eps) or eps <= 0:
        eps = 0.01
    return max(1, int(math.ceil(1.0 / eps)))


def yu_basic_oracle_depth(ds_info: DatasetInfo, qp: QuantumParams) -> int:
    """Depth proxy for one basic database oracle O: |i>|j>|a> -> |i>|j>|a xor D_ij>."""
    if str(qp.yu_basic_oracle_depth_model).lower() == "unit":
        return 1
    return max(1, log2ceil(ds_info.n_transactions) + log2ceil(ds_info.n_items))


def qfm_preprocessing_depth(ds_info: DatasetInfo, stats: Dict[str, Any], qp: QuantumParams) -> int:
    """QPr/cache-initialization proxy. Default nm is conservative and comparable to Yu."""
    M, N = ds_info.n_transactions, ds_info.n_items
    model = str(qp.qfm_preprocess_model).lower()
    if model == "qpr_qcount":
        # Approximate QPr support detection with quantum counting over transactions for each item.
        # This does not include detailed physical QRAM write costs; use nm for conservative comparisons.
        return int(math.ceil(qp.c_pre_qfm * N * math.ceil(math.sqrt(max(1, M))) * (log2ceil(N) + log2ceil(M))))
    return int(math.ceil(qp.c_pre_qfm * N * M))


def qfm_cache_update_depth(t_k: int, M: int, qp: QuantumParams) -> int:
    model = str(qp.qfm_cache_update_model).lower()
    if model == "none":
        return 0
    if model == "linear_m":
        return int(max(0, t_k) * max(1, M))
    # logm: optimistic QRAM/cache-addressing model
    return int(math.ceil(max(0, t_k) * log2ceil(M)))


def estimate_yu_full(ds_info: DatasetInfo, stats: Dict[str, Any], qp: QuantumParams, num_queries: int = 1) -> Dict[str, Any]:
    M, N = ds_info.n_transactions, ds_info.n_items
    len_hist = get_int_dict(stats.get("len_hist", {}))
    observed_max_k = max(len_hist.keys()) if len_hist else 0
    max_k = int(qp.yu_max_k) if int(qp.yu_max_k) > 0 else observed_max_k
    max_k = max(0, max_k)
    pre_depth = int(math.ceil(qp.c_pre_yu * N * M))
    logN = log2ceil(N); logM = log2ceil(M)
    body_one_query = 0
    levels: List[Dict[str, Any]] = []

    if str(qp.yu_model).lower() == "old_additive":
        # Legacy model kept only for sensitivity/debug comparisons.
        for k in range(1, max_k + 1):
            t_k = int(len_hist.get(k, 0))
            Ck = candidate_count_for_k(k, N, len_hist, stats)
            logC = log2ceil(Ck)
            stateprep = int(math.ceil(qp.c_yu_stateprep_logc * logC + qp.c_yu_stateprep_k * k))
            support_eval_oracle = int(math.ceil(qp.c_yu_oracle_lognm * (logN + logM) + qp.c_yu_mark_logm * logM + qp.c_yu_mark_logc * logC))
            grover_op_depth = int(stateprep + support_eval_oracle + math.ceil(qp.c_yu_diffusion_logc * logC))
            qae_reps = qae_repetitions(M, qp.qae_mode)
            qae_depth = int(qae_reps * grover_op_depth)
            listing_queries = grover_findall_queries(Ck, t_k)
            listing_depth = int(listing_queries * grover_op_depth)
            level_depth = int(qae_depth + listing_depth)
            body_one_query += level_depth
            levels.append(dict(k=k, candidates=Ck, frequent=t_k, yu_model="old_additive", stateprep_depth=stateprep, support_eval_oracle_depth=support_eval_oracle, grover_op_depth=grover_op_depth, qae_repetitions=qae_reps, qae_depth=qae_depth, listing_queries=listing_queries, listing_depth=listing_depth, level_depth=level_depth))
        total_depth = int(pre_depth + body_one_query * int(num_queries))
        return dict(method="Yu", dataset=ds_info.dataset, preprocessing_depth=pre_depth, body_one_query_depth=body_one_query, total_depth=total_depth, runtime_sec=float(total_depth) * float(qp.gate_time_ns) * 1e-9, max_k=max_k, levels=levels, model="Yu old additive proxy: preprocessing + per-level QAE + Grover listing")

    # Paper-aligned qARM/Yu model.
    # Yu uses parallel amplitude estimation with T=Theta(1/epsilon), then amplitude amplification.
    # The cost is multiplicative: sqrt(C_k * F_k) * T * depth(G^(k)), not additive T + sqrt(CF).
    qae_reps = yu_qae_repetitions_paper(qp.yu_epsilon)
    basic_depth = yu_basic_oracle_depth(ds_info, qp)
    for k in range(1, max_k + 1):
        t_k = int(len_hist.get(k, 0))
        Ck = candidate_count_for_k(k, N, len_hist, stats)
        logC = log2ceil(Ck)
        listing_factor = grover_findall_queries(Ck, t_k)  # ~= sqrt(C_k F_k), capped by C_k
        # O^(k) uses 2k basic database oracle calls + Theta(k) gates in Yu/qARM.
        ok_depth = int(math.ceil(2 * int(k) * basic_depth + qp.c_yu_stateprep_k * int(k)))
        transaction_diffusion_depth = int(math.ceil(qp.c_yu_mark_logm * logM))
        candidate_diffusion_depth = int(math.ceil(qp.c_yu_diffusion_logc * logC))
        grover_op_depth = int(ok_depth + transaction_diffusion_depth + candidate_diffusion_depth)
        # Candidate state generation via QRAM-style O_C; repeated under amplitude amplification by default.
        candidate_prep_depth = int(math.ceil(max(1, k) * log2ceil(max(1, N * max(1, Ck)))))
        if str(qp.yu_candidate_prep_model).lower() == "none":
            candidate_prep_total = 0
        elif str(qp.yu_candidate_prep_model).lower() == "once":
            candidate_prep_total = candidate_prep_depth
        else:
            candidate_prep_total = int(listing_factor * candidate_prep_depth)
        qae_depth_per_listing = int(qae_reps * grover_op_depth)
        listing_qae_depth = int(listing_factor * qae_depth_per_listing)
        level_depth = int(candidate_prep_total + listing_qae_depth)
        body_one_query += level_depth
        levels.append(dict(
            k=k,
            candidates=Ck,
            frequent=t_k,
            yu_model="paper",
            epsilon=float(qp.yu_epsilon),
            basic_oracle_depth=basic_depth,
            O_k_depth=ok_depth,
            qae_repetitions=qae_reps,
            listing_factor=listing_factor,
            candidate_prep_depth=candidate_prep_depth,
            candidate_prep_total_depth=candidate_prep_total,
            grover_op_depth=grover_op_depth,
            qae_depth_per_listing=qae_depth_per_listing,
            listing_qae_depth=listing_qae_depth,
            level_depth=level_depth,
            basic_oracle_calls=int(2 * int(k) * qae_reps * listing_factor),
        ))
    total_depth = int(pre_depth + body_one_query * int(num_queries))
    return dict(
        method="Yu",
        dataset=ds_info.dataset,
        preprocessing_depth=pre_depth,
        body_one_query_depth=body_one_query,
        total_depth=total_depth,
        runtime_sec=float(total_depth) * float(qp.gate_time_ns) * 1e-9,
        max_k=max_k,
        levels=levels,
        model=f"Yu paper-aligned proxy: preprocessing + sum_k sqrt(Ck*Fk)*ceil(1/epsilon)*depth(G^k); epsilon={float(qp.yu_epsilon)}; basic_oracle={qp.yu_basic_oracle_depth_model}",
    )

def estimate_qfm_full(ds_info: DatasetInfo, stats: Dict[str, Any], qp: QuantumParams, num_queries: int = 1) -> Dict[str, Any]:
    M, N = ds_info.n_transactions, ds_info.n_items
    len_hist = get_int_dict(stats.get("len_hist", {}))
    observed_max_k = max(len_hist.keys()) if len_hist else 0
    max_k = int(qp.qfm_max_k) if int(qp.qfm_max_k) > 0 else observed_max_k
    max_k = max(0, max_k)
    logM = log2ceil(M)
    # QPr/cache initialization. By default nm is conservative and comparable with Yu preprocessing.
    pre_depth = qfm_preprocessing_depth(ds_info, stats, qp)
    body_one_query = 0
    levels: List[Dict[str, Any]] = []
    start_k = max(1, int(qp.qfm_start_k))
    if start_k > 1 and max_k >= 1:
        levels.append(dict(k=1, candidates=candidate_count_for_k(1, N, len_hist, stats), frequent=int(len_hist.get(1, 0)), status="absorbed_into_QPr", oracle_depth=0, grover_op_depth=0, listing_queries=0, listing_depth=0, cache_update_depth=0, level_depth=0))
    for k in range(start_k, max_k + 1):
        t_k = int(len_hist.get(k, 0))
        Ck = candidate_count_for_k(k, N, len_hist, stats)
        logC = log2ceil(Ck)
        prev_L = max(1, int(len_hist.get(k - 1, 1)))
        qram = int(math.ceil(qp.c_qfm_qram_log * log2ceil(prev_L)))
        bitparallel = int(math.ceil(qp.c_qfm_bitparallel * logM))
        if str(getattr(qp, "qfm_popcount_model", "logm")).lower() == "logm2":
            popcount = int(math.ceil(qp.c_qfm_popcount * (logM ** 2)))
        else:
            popcount = int(math.ceil(qp.c_qfm_popcount * logM))
        cmpd = int(math.ceil(qp.c_qfm_cmp * logM))
        oracle = int(qram + bitparallel + popcount + cmpd)
        grover_op_depth = int(oracle + math.ceil(qp.c_qfm_diffusion_logc * logC))
        listing_queries = grover_findall_queries(Ck, t_k)
        listing_depth = int(listing_queries * grover_op_depth)
        cache_update = qfm_cache_update_depth(t_k, M, qp)
        level_depth = int(listing_depth + cache_update)
        body_one_query += level_depth
        levels.append(dict(k=k, candidates=Ck, frequent=t_k, status="ok", oracle_depth=oracle, qram_depth=qram, bitparallel_depth=bitparallel, popcount_depth=popcount, comparison_depth=cmpd, grover_op_depth=grover_op_depth, listing_queries=listing_queries, listing_depth=listing_depth, cache_update_model=qp.qfm_cache_update_model, cache_update_depth=cache_update, level_depth=level_depth))
    total_depth = int(pre_depth + body_one_query * int(num_queries))
    return dict(
        method="QFM",
        dataset=ds_info.dataset,
        preprocessing_depth=pre_depth,
        body_one_query_depth=body_one_query,
        total_depth=total_depth,
        runtime_sec=float(total_depth) * float(qp.gate_time_ns) * 1e-9,
        max_k=max_k,
        levels=levels,
        model=f"QFM proxy: {qp.qfm_preprocess_model} preprocessing; body starts k={start_k}; cached threshold oracle + Grover listing + {qp.qfm_cache_update_model} cache update",
    )


# -----------------------------
# Search and orchestration
# -----------------------------

def support_count_from_percent(M: int, percent: float) -> int:
    return int(math.ceil(float(percent) / 100.0 * max(1, int(M))))


def compute_score(qfm: Dict[str, Any], classical: Dict[str, Dict[str, Any]], stats: Dict[str, Any], args: argparse.Namespace) -> Tuple[float, Dict[str, Any]]:
    qfm_t = safe_float(qfm.get("runtime_sec"), 0.0)
    len_hist = get_int_dict(stats.get("len_hist", {}))
    L2 = int(len_hist.get(2, 0))
    max_k = safe_int(stats.get("max_itemset_len"), 0)
    ok_times: List[float] = []
    timeout_count = 0
    for alg, rec in classical.items():
        if rec.get("status") == "timeout":
            timeout_count += 1
            # Timeout means at least this large; count it as timeout_sec for scoring.
            ok_times.append(float(args.timeout_sec))
        elif rec.get("status") == "ok":
            t = rec.get("runtime_sec")
            if t is not None:
                ok_times.append(float(t))
    median_classical = float(sorted(ok_times)[len(ok_times)//2]) if ok_times else 0.0
    min_classical = float(min(ok_times)) if ok_times else 0.0
    adv_median = (median_classical / qfm_t) if qfm_t > 0 else 0.0
    adv_min = (min_classical / qfm_t) if qfm_t > 0 else 0.0
    trivial_penalty = 0.0
    if L2 <= 0:
        trivial_penalty += 1000.0
    if max_k < 2:
        trivial_penalty += 1000.0
    if L2 < int(args.min_l2):
        trivial_penalty += 10.0
    timeout_penalty = timeout_count * float(args.timeout_penalty)
    # Min-support too high can be seen as degenerate. Penalize high percent softly.
    high_minsup_penalty = max(0.0, safe_float(stats.get("minsup_percent"), 0.0) - float(args.high_minsup_penalty_after)) * 0.01
    score = adv_median + 0.25 * adv_min - trivial_penalty - timeout_penalty - high_minsup_penalty
    detail = dict(
        qfm_runtime_sec=qfm_t,
        median_classical_runtime_sec=median_classical,
        min_classical_runtime_sec=min_classical,
        advantage_vs_median=adv_median,
        advantage_vs_min=adv_min,
        timeout_count=timeout_count,
        L2=L2,
        max_k=max_k,
        score=score,
    )
    return score, detail


def run_candidate(ds_info: DatasetInfo, percent: float, tools: ToolPaths, qp: QuantumParams, args: argparse.Namespace) -> Dict[str, Any]:
    minsup_count = support_count_from_percent(ds_info.n_transactions, percent)
    # First run a normal full FIM miner to obtain len_hist and stats for all levels. Eclat is usually robust.
    stats_rec = run_spmf_stats(args.stats_algorithm, ds_info, percent, minsup_count, tools, Path(args.results_dir), args.timeout_sec, args.force)
    stats = dict(stats_rec)
    stats["minsup_percent"] = float(percent)
    stats["minsup_count"] = int(minsup_count)
    qfm = estimate_qfm_full(ds_info, stats, qp, num_queries=int(args.num_queries))
    yu = estimate_yu_full(ds_info, stats, qp, num_queries=int(args.num_queries))

    classical: Dict[str, Dict[str, Any]] = {}
    if "FP-Growth" in args.classical_methods:
        classical["FP-Growth"] = run_spmf_stats("FPGrowth_itemsets", ds_info, percent, minsup_count, tools, Path(args.results_dir), args.timeout_sec, args.force)
    if "Eclat" in args.classical_methods:
        # Reuse stats if stats_algorithm == Eclat, otherwise run separately.
        if args.stats_algorithm == "Eclat":
            tmp = dict(stats_rec); tmp["algorithm"] = "Eclat"
            classical["Eclat"] = tmp
        else:
            classical["Eclat"] = run_spmf_stats("Eclat", ds_info, percent, minsup_count, tools, Path(args.results_dir), args.timeout_sec, args.force)
    if "Hamm" in args.classical_methods:
        classical["Hamm"] = run_hamm(ds_info, percent, minsup_count, tools, Path(args.results_dir), args.timeout_sec, args.force)
    if "CICLAD" in args.classical_methods:
        classical["CICLAD"] = run_ciclad(ds_info, minsup_count, tools, Path(args.results_dir), args.timeout_sec, args.force)

    score, score_detail = compute_score(qfm, classical, stats, args)
    return dict(
        dataset=ds_info.dataset,
        minsup_percent=float(percent),
        minsup_count=int(minsup_count),
        n_transactions=ds_info.n_transactions,
        n_items=ds_info.n_items,
        stats=stats,
        qfm=qfm,
        yu=yu,
        classical=classical,
        score=score,
        score_detail=score_detail,
    )


def make_pruned_timeout_record(ds_info: DatasetInfo, alg: str, percent: float, minsup_count: int, args: argparse.Namespace, *, source_timeout_percent: float, reason: str) -> Dict[str, Any]:
    return dict(
        dataset=ds_info.dataset,
        algorithm=alg,
        status="pruned_timeout_monotonic",
        runtime_sec=float(args.timeout_sec),
        timeout_sec=float(args.timeout_sec),
        elapsed_at_timeout_sec=None,
        minsup_percent=float(percent),
        minsup_count=int(minsup_count),
        minsup_count_filter=int(minsup_count),
        pattern_count=0,
        max_itemset_len=0,
        len_hist={},
        union_sizes={},
        join_cand={},
        error=reason,
        predicted_from_timeout_at_minsup_percent=float(source_timeout_percent),
        note="Skipped by monotonic timeout pruning: lower minsup is expected to be no easier than an already timed-out higher-minsup point.",
    )


def should_prune_algorithm(alg: str, percent: float, prune_state: Dict[str, Any], args: argparse.Namespace) -> Optional[float]:
    if not getattr(args, "monotonic_timeout_prune", False):
        return None
    timeout_map = prune_state.setdefault("algorithm_timeout_at", {})
    t = timeout_map.get(alg)
    if t is None:
        return None
    # Candidate percentages are evaluated from high to low. A smaller/equal minsup is at least as hard.
    if float(percent) <= float(t):
        return float(t)
    return None


def run_candidate_optimized(ds_info: DatasetInfo, percent: float, tools: ToolPaths, qp: QuantumParams, args: argparse.Namespace, prune_state: Dict[str, Any]) -> Dict[str, Any]:
    minsup_count = support_count_from_percent(ds_info.n_transactions, percent)
    results_path = Path(args.results_dir)

    # The statistics miner is required for len_hist and for full-level QFM/Yu estimates.
    # If it times out at some support, lower supports are pruned by the caller.
    stats_rec = run_spmf_stats(args.stats_algorithm, ds_info, percent, minsup_count, tools, results_path, args.timeout_sec, args.force)
    stats = dict(stats_rec)
    stats["minsup_percent"] = float(percent)
    stats["minsup_count"] = int(minsup_count)

    if stats_rec.get("status") != "ok":
        if stats_rec.get("status") == "timeout":
            prune_state["stats_timeout_at"] = float(percent)
        # Return a degenerate row so the audit file records where pruning/error began.
        reason = f"not estimated: stats miner status={stats_rec.get('status')}"
        empty_qfm = dict(method="QFM", dataset=ds_info.dataset, preprocessing_depth=0, body_one_query_depth=0, total_depth=0, runtime_sec=0.0, max_k=0, levels=[], model=reason)
        empty_yu = dict(method="Yu", dataset=ds_info.dataset, preprocessing_depth=0, body_one_query_depth=0, total_depth=0, runtime_sec=0.0, max_k=0, levels=[], model=reason)
        classical = {args.stats_algorithm: dict(stats_rec)}
        score, score_detail = compute_score(empty_qfm, classical, stats, args)
        return dict(dataset=ds_info.dataset, minsup_percent=float(percent), minsup_count=int(minsup_count), n_transactions=ds_info.n_transactions, n_items=ds_info.n_items, stats=stats, qfm=empty_qfm, yu=empty_yu, classical=classical, score=score - 10000.0, score_detail=score_detail)

    qfm = estimate_qfm_full(ds_info, stats, qp, num_queries=int(args.num_queries))
    yu = estimate_yu_full(ds_info, stats, qp, num_queries=int(args.num_queries))

    classical: Dict[str, Dict[str, Any]] = {}
    jobs: List[Tuple[str, Any]] = []

    def already_or_submit(alg: str):
        pruned_at = should_prune_algorithm(alg, percent, prune_state, args)
        if pruned_at is not None:
            classical[alg] = make_pruned_timeout_record(ds_info, alg, percent, minsup_count, args, source_timeout_percent=pruned_at, reason=f"{alg} timed out previously at minsup={pruned_at}%")
            return
        if alg == "FP-Growth":
            if args.stats_algorithm == "FPGrowth_itemsets":
                tmp = dict(stats_rec); tmp["algorithm"] = "FP-Growth"
                classical[alg] = tmp
            else:
                jobs.append((alg, lambda: run_spmf_stats("FPGrowth_itemsets", ds_info, percent, minsup_count, tools, results_path, args.timeout_sec, args.force)))
        elif alg == "Eclat":
            if args.stats_algorithm == "Eclat":
                tmp = dict(stats_rec); tmp["algorithm"] = "Eclat"
                classical[alg] = tmp
            else:
                jobs.append((alg, lambda: run_spmf_stats("Eclat", ds_info, percent, minsup_count, tools, results_path, args.timeout_sec, args.force)))
        elif alg == "Hamm":
            jobs.append((alg, lambda: run_hamm(ds_info, percent, minsup_count, tools, results_path, args.timeout_sec, args.force)))
        elif alg == "CICLAD":
            jobs.append((alg, lambda: run_ciclad(ds_info, minsup_count, tools, results_path, args.timeout_sec, args.force)))

    for alg in args.classical_methods:
        already_or_submit(alg)

    if jobs:
        if int(getattr(args, "baseline_jobs", 1)) <= 1:
            for alg, fn in jobs:
                try:
                    classical[alg] = fn()
                except Exception as e:
                    classical[alg] = dict(
                        dataset=ds_info.dataset, algorithm=alg, status="error", runtime_sec=None,
                        minsup_percent=float(percent), minsup_count=int(minsup_count),
                        pattern_count=0, max_itemset_len=0, len_hist={}, union_sizes={}, join_cand={},
                        error=f"{type(e).__name__}: {e}",
                    )
        else:
            with ThreadPoolExecutor(max_workers=int(args.baseline_jobs)) as ex:
                futs = {ex.submit(fn): alg for alg, fn in jobs}
                for fut in as_completed(futs):
                    alg = futs[fut]
                    try:
                        classical[alg] = fut.result()
                    except Exception as e:
                        classical[alg] = dict(
                            dataset=ds_info.dataset, algorithm=alg, status="error", runtime_sec=None,
                            minsup_percent=float(percent), minsup_count=int(minsup_count),
                            pattern_count=0, max_itemset_len=0, len_hist={}, union_sizes={}, join_cand={},
                            error=f"{type(e).__name__}: {e}",
                        )

    if getattr(args, "monotonic_timeout_prune", False):
        timeout_map = prune_state.setdefault("algorithm_timeout_at", {})
        for alg, rec in classical.items():
            if rec.get("status") == "timeout":
                # Once a method times out at this support threshold, skip it for lower supports.
                timeout_map.setdefault(alg, float(percent))

    score, score_detail = compute_score(qfm, classical, stats, args)
    return dict(
        dataset=ds_info.dataset,
        minsup_percent=float(percent),
        minsup_count=int(minsup_count),
        n_transactions=ds_info.n_transactions,
        n_items=ds_info.n_items,
        stats=stats,
        qfm=qfm,
        yu=yu,
        classical=classical,
        score=score,
        score_detail=score_detail,
    )


def evaluate_dataset_optimized(ds: str, ds_info: DatasetInfo, candidates: List[float], tools: ToolPaths, qp: QuantumParams, args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    # Monotonic pruning requires evaluating from larger minsup to smaller minsup.
    ordered = sorted([float(x) for x in candidates], reverse=True)
    rows: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None
    prune_state: Dict[str, Any] = {"algorithm_timeout_at": {}}

    for pct in ordered:
        if prune_state.get("stats_timeout_at") is not None and getattr(args, "monotonic_timeout_prune", False):
            print(f"[prune][{ds}] stats miner timed out at {prune_state['stats_timeout_at']}%; skip remaining lower minsup candidates.")
            break
        print(f"[candidate] {ds} minsup={pct}%")
        row = run_candidate_optimized(ds_info, pct, tools, qp, args, prune_state)
        rows.append(row)
        if best is None or float(row["score"]) > float(best["score"]):
            best = row
        sd = row["score_detail"]
        if args.stop_first_good and sd.get("L2", 0) >= int(args.min_l2) and sd.get("advantage_vs_median", 0) > 1.0:
            break
    return rows, best


def flatten_candidate(row: Dict[str, Any]) -> Dict[str, Any]:
    stats = row["stats"]
    sd = row["score_detail"]
    out = dict(
        dataset=row["dataset"],
        minsup_percent=row["minsup_percent"],
        minsup_count=row["minsup_count"],
        n_transactions=row["n_transactions"],
        n_items=row["n_items"],
        max_itemset_len=stats.get("max_itemset_len"),
        pattern_count=stats.get("pattern_count"),
        L1=get_int_dict(stats.get("len_hist", {})).get(1, 0),
        L2=get_int_dict(stats.get("len_hist", {})).get(2, 0),
        QFM_runtime_sec=row["qfm"].get("runtime_sec"),
        Yu_runtime_sec=row["yu"].get("runtime_sec"),
        QFM_total_depth=row["qfm"].get("total_depth"),
        Yu_total_depth=row["yu"].get("total_depth"),
        QFM_model=row["qfm"].get("model"),
        Yu_model=row["yu"].get("model"),
        score=row["score"],
        advantage_vs_median=sd.get("advantage_vs_median"),
        advantage_vs_min=sd.get("advantage_vs_min"),
        timeout_count=sd.get("timeout_count"),
    )
    for alg, rec in row.get("classical", {}).items():
        key = alg.replace("-", "_").replace(" ", "_")
        out[f"{key}_runtime_sec"] = rec.get("runtime_sec")
        out[f"{key}_status"] = rec.get("status")
        out[f"{key}_pattern_count"] = rec.get("pattern_count")
    return out


def final_rows_for_selected(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    common = dict(
        dataset=row["dataset"],
        minsup_percent=row["minsup_percent"],
        minsup_count=row["minsup_count"],
        n_transactions=row["n_transactions"],
        n_items=row["n_items"],
        max_itemset_len=row["stats"].get("max_itemset_len"),
        L1=get_int_dict(row["stats"].get("len_hist", {})).get(1, 0),
        L2=get_int_dict(row["stats"].get("len_hist", {})).get(2, 0),
    )
    for q in [row["qfm"], row["yu"]]:
        rows.append({**common, "method": q["method"], "source": "quantum_proxy", "status": "ok", "runtime_sec": q["runtime_sec"], "total_depth": q["total_depth"], "preprocessing_depth": q["preprocessing_depth"], "body_one_query_depth": q["body_one_query_depth"], "max_k_model": q["max_k"], "note": q["model"]})
    for alg, rec in row.get("classical", {}).items():
        rows.append({**common, "method": alg, "source": "classical_wallclock", "status": rec.get("status"), "runtime_sec": rec.get("runtime_sec"), "total_depth": None, "preprocessing_depth": None, "body_one_query_depth": None, "max_k_model": None, "note": rec.get("error", rec.get("cmd", ""))})
    return rows


def quantum_rows_for_point(ds_info: DatasetInfo, stats: Dict[str, Any], qp: QuantumParams, *, dataset: str, sweep_type: str, tx_ratio: float, minsup_percent: float, minsup_count: int, num_queries: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    qfm = estimate_qfm_full(ds_info, stats, qp, num_queries=int(num_queries))
    yu = estimate_yu_full(ds_info, stats, qp, num_queries=int(num_queries))
    rows: List[Dict[str, Any]] = []
    level_rows: List[Dict[str, Any]] = []
    for q in [qfm, yu]:
        rows.append(dict(
            dataset=dataset,
            sweep_type=sweep_type,
            tx_ratio_percent=float(tx_ratio),
            minsup_percent=float(minsup_percent),
            minsup_count=int(minsup_count),
            n_transactions=int(ds_info.n_transactions),
            n_items=int(ds_info.n_items),
            method=q["method"],
            preprocessing_depth=q["preprocessing_depth"],
            body_one_query_depth=q["body_one_query_depth"],
            total_depth=q["total_depth"],
            runtime_sec=q["runtime_sec"],
            max_k=q["max_k"],
            model=q["model"],
            stats_max_itemset_len=stats.get("max_itemset_len"),
            stats_pattern_count=stats.get("pattern_count"),
            L1=get_int_dict(stats.get("len_hist", {})).get(1, 0),
            L2=get_int_dict(stats.get("len_hist", {})).get(2, 0),
            status="ok",
        ))
        for lr in q.get("levels", []):
            level_rows.append(dict(dataset=dataset, sweep_type=sweep_type, tx_ratio_percent=float(tx_ratio), minsup_percent=float(minsup_percent), method=q["method"], **lr))
    return rows, level_rows


def run_table12(ds_infos: Dict[str, DatasetInfo], table12_datasets: List[str], tools: ToolPaths, qp: QuantumParams, args: argparse.Namespace, results_dir: Path) -> None:
    """Run Table 1/2 quantum full-level sweeps: tx-ratio sweep and minsup sweep."""
    tx_ratios = parse_csv_floats(args.tx_ratios)
    minsup_ratios = parse_csv_floats(args.minsup_ratios)
    override = parse_override_default_minsup(args.override_default_minsup)
    table12_dir = ensure_dir(results_dir / "table12_work")
    tx_rows: List[Dict[str, Any]] = []
    ms_rows: List[Dict[str, Any]] = []
    level_rows: List[Dict[str, Any]] = []

    def default_minsup(ds: str) -> float:
        # Preserve old runner defaults, overridden by CLI map.
        defaults = {"mushroom": 1.0, "connect4": 1.0, "chess": 1.0, "tic_tac_toe": 2.0, "car": 5.0, "nursery": 2.0}
        return float(override.get(ds, defaults.get(ds, 1.0)))

    jobs: List[Tuple[str, str, float, Any]] = []
    # We parallelize individual table12 points because there is no timeout-pruning dependency here.
    def make_tx_job(ds: str, r: float):
        def _run():
            base = ds_infos[ds]
            sub_dir = ensure_dir(table12_dir / ds)
            sub_spmf = sub_dir / f"sub_tx{float(r):.12g}.spmf"
            sub_dat = sub_dir / f"sub_tx{float(r):.12g}.dat"
            if (not sub_spmf.exists()) or args.force:
                n_sub = subsample_spmf_lines(base.spmf_path, str(sub_spmf), float(r), seed=int(args.random_seed))
                shutil.copyfile(sub_spmf, sub_dat)
            else:
                with open(sub_spmf, "r", encoding="utf-8", errors="ignore") as f:
                    n_sub = sum(1 for ln in f if ln.strip())
            sub_info = dataset_info_for_spmf_subset(base, sub_spmf, sub_dat, n_sub, suffix=f"tx{float(r):.12g}")
            ms_default = default_minsup(ds)
            if args.tx_sweep_minsup_mode == "count":
                fixed_count = int(math.ceil(ms_default / 100.0 * max(1, base.n_transactions)))
                eff_percent = 100.0 * float(fixed_count) / float(max(1, n_sub))
                # Run at effective percent, then post-filter exact absolute support count.
                stats = run_spmf_stats(args.stats_algorithm, sub_info, eff_percent, fixed_count, tools, results_dir, args.timeout_sec, args.force)
                minsup_percent_for_record = float(ms_default)
                minsup_count = int(fixed_count)
            else:
                minsup_count = int(math.ceil(ms_default / 100.0 * max(1, n_sub)))
                stats = run_spmf_stats(args.stats_algorithm, sub_info, ms_default, None, tools, results_dir, args.timeout_sec, args.force)
                minsup_percent_for_record = float(ms_default)
            rows, lv = quantum_rows_for_point(sub_info, stats, qp, dataset=ds, sweep_type="tx_sweep", tx_ratio=float(r), minsup_percent=minsup_percent_for_record, minsup_count=minsup_count, num_queries=int(args.num_queries))
            # Add effective percent details for count mode.
            for row in rows:
                row["tx_sweep_minsup_mode"] = args.tx_sweep_minsup_mode
                row["effective_minsup_percent"] = float(100.0 * minsup_count / max(1, n_sub))
                row["full_dataset_default_minsup_percent"] = float(ms_default)
            return rows, lv
        return _run

    def make_ms_job(ds: str, ms: float):
        def _run():
            base = ds_infos[ds]
            minsup_count = int(math.ceil(float(ms) / 100.0 * max(1, base.n_transactions)))
            stats = run_spmf_stats(args.stats_algorithm, base, float(ms), None, tools, results_dir, args.timeout_sec, args.force)
            rows, lv = quantum_rows_for_point(base, stats, qp, dataset=ds, sweep_type="minsup_sweep", tx_ratio=100.0, minsup_percent=float(ms), minsup_count=minsup_count, num_queries=int(args.num_queries))
            for row in rows:
                row["tx_sweep_minsup_mode"] = "not_applicable"
                row["effective_minsup_percent"] = float(ms)
                row["full_dataset_default_minsup_percent"] = default_minsup(ds)
            return rows, lv
        return _run

    for ds in table12_datasets:
        for r in tx_ratios:
            jobs.append((ds, "tx", float(r), make_tx_job(ds, float(r))))
        for ms in minsup_ratios:
            jobs.append((ds, "minsup", float(ms), make_ms_job(ds, float(ms))))

    print(f"[table12] running {len(jobs)} quantum sweep points with jobs={args.table12_jobs}")
    if int(args.table12_jobs) <= 1:
        for ds, kind, val, fn in jobs:
            try:
                rows, lv = fn()
                if kind == "tx": tx_rows.extend(rows)
                else: ms_rows.extend(rows)
                level_rows.extend(lv)
            except Exception as e:
                print(f"[error][table12][{ds}][{kind}={val}] {type(e).__name__}: {e}", file=sys.stderr)
    else:
        with ThreadPoolExecutor(max_workers=int(args.table12_jobs)) as ex:
            futs = {ex.submit(fn): (ds, kind, val) for ds, kind, val, fn in jobs}
            for fut in as_completed(futs):
                ds, kind, val = futs[fut]
                try:
                    rows, lv = fut.result()
                    if kind == "tx": tx_rows.extend(rows)
                    else: ms_rows.extend(rows)
                    level_rows.extend(lv)
                except Exception as e:
                    print(f"[error][table12][{ds}][{kind}={val}] {type(e).__name__}: {e}", file=sys.stderr)

    tx_rows.sort(key=lambda r: (r["dataset"], r["tx_ratio_percent"], r["method"]))
    ms_rows.sort(key=lambda r: (r["dataset"], r["minsup_percent"], r["method"]))
    level_rows.sort(key=lambda r: (r["dataset"], r["sweep_type"], r.get("tx_ratio_percent", 0), r.get("minsup_percent", 0), r["method"], r.get("k", 0)))
    write_csv(results_dir / "table1_tx_sweep_quantum_full.csv", tx_rows)
    write_csv(results_dir / "table2_minsup_sweep_quantum_full.csv", ms_rows)
    write_csv(results_dir / "table12_levels_quantum_full.csv", level_rows)
    print(f"[done][table12] wrote {results_dir / 'table1_tx_sweep_quantum_full.csv'}")
    print(f"[done][table12] wrote {results_dir / 'table2_minsup_sweep_quantum_full.csv'}")


def parse_csv_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description="All-in-one QFM/Yu full-level and Table 5 baseline experiment driver.")
    ap.add_argument("--mode", choices=["table12", "table5", "all"], default="all",
                    help="table12: original tx/minsup sweeps for QFM/Yu; table5: large-dataset selected-minsup baselines; all: run both.")
    ap.add_argument("--datasets", default="accidents,pumsb,pumsb_star", help="Table-5 datasets (large datasets).")
    ap.add_argument("--table12-datasets", default="mushroom,connect4,chess,tic_tac_toe,car,nursery", help="Table-1/2 datasets for tx/minsup sweeps.")
    ap.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "../../qcount_1128/data"), help="Data dir for Table 5 large/FIMI datasets.")
    ap.add_argument("--table12-data-dir", default=os.environ.get("TABLE12_DATA_DIR", ""), help="Optional data dir for Table 1/2 datasets. If empty, uses --data-dir.")
    ap.add_argument("--results-dir", default="results_table_allinone")
    ap.add_argument("--project-dir", default=os.getcwd())
    ap.add_argument("--candidate-minsup", default="99.8,99.5,99,98,95,90,80,70,60,50", help="Table-5 candidate minsup percentages searched high-to-low.")
    # Old experiment_parallel-style parameters for Table 1/2 sweeps.
    ap.add_argument("--tx-ratios", default="10,20,30,40,50,60,70", help="Table-1 tx sweep ratios, old runner style.")
    ap.add_argument("--minsup-ratios", default="10,20,30,40,50,60,70", help="Table-2 minsup sweep ratios, old runner style.")
    ap.add_argument("--override-default-minsup", default="mushroom=10,connect4=10,tic-tac-toe=10,car=10,kr-vs-kp=10,nursery=10", help="Old runner style default minsup map for tx sweep.")
    ap.add_argument("--tx-sweep-minsup-mode", choices=["percent", "count"], default="count", help="For Table-1 tx sweep: percent scales with subsample; count fixes support count from full dataset.")
    ap.add_argument("--random-seed", type=int, default=42)
    ap.add_argument("--table12-jobs", type=int, default=1, help="Parallel Table-1/2 sweep points. Keep modest because each point runs SPMF stats miner.")
    ap.add_argument("--stop-first-good", action="store_true", help="Stop per dataset after first candidate with QFM faster than median classical and L2 constraint satisfied.")
    ap.add_argument("--min-l2", type=int, default=1)
    ap.add_argument("--timeout-sec", type=int, default=1800)
    ap.add_argument("--timeout-penalty", type=float, default=0.25, help="Score penalty per timed-out baseline. Set 0 to not penalize CICLAD timeout.")
    ap.add_argument("--high-minsup-penalty-after", type=float, default=99.0)
    ap.add_argument("--jobs", type=int, default=1, help="Parallel datasets only in optimized mode. This avoids launching all minsup candidates before pruning can take effect.")
    ap.add_argument("--baseline-jobs", type=int, default=1, help="Parallel external baselines within one dataset/minsup candidate. Keep small to avoid oversubscription.")
    ap.add_argument("--monotonic-timeout-prune", action=argparse.BooleanOptionalAction, default=True, help="If a method times out at a minsup, skip that method for lower minsup candidates. Valid when candidate minsup is evaluated high-to-low.")
    ap.add_argument("--force", action="store_true", help="Ignore cached records.")
    ap.add_argument("--classical-methods", default="FP-Growth,Eclat,Hamm,CICLAD")
    ap.add_argument("--stats-algorithm", choices=["Eclat", "FPGrowth_itemsets"], default="FPGrowth_itemsets")
    ap.add_argument("--spmf-jar", default="")
    ap.add_argument("--java-cmd", default="")
    ap.add_argument("--hamm-bin", default="")
    ap.add_argument("--ciclad-bin", default="")
    ap.add_argument("--gate-time-ns", type=float, default=25.0)
    ap.add_argument("--qae-mode", choices=["sqrtm", "linear"], default="sqrtm", help="Legacy Yu additive model only.")
    ap.add_argument("--num-queries", type=int, default=1)
    ap.add_argument("--yu-max-k", type=int, default=0)
    ap.add_argument("--qfm-max-k", type=int, default=0)
    ap.add_argument("--yu-model", choices=["paper", "old_additive"], default="paper", help="paper uses Yu/qARM multiplicative T*sqrt(CF) formula; old_additive keeps the previous optimistic proxy.")
    ap.add_argument("--yu-epsilon", type=float, default=0.01, help="Amplitude-estimation error epsilon for Yu paper model; T=ceil(1/epsilon).")
    ap.add_argument("--yu-basic-oracle-depth-model", choices=["unit", "lognm"], default="lognm")
    ap.add_argument("--yu-candidate-prep-model", choices=["repeated", "once", "none"], default="repeated")
    ap.add_argument("--qfm-start-k", type=int, default=2, help="Default 2: QPr absorbs L1; body starts from k=2.")
    ap.add_argument("--qfm-cache-update-model", choices=["logm", "linear_m", "none"], default="logm")
    ap.add_argument("--qfm-preprocess-model", choices=["nm", "qpr_qcount"], default="nm")
    ap.add_argument("--qfm-popcount-model", choices=["logm", "logm2"], default="logm", help="QFM popcount/comparator depth model. logm matches the current paper-level Lemma; logm2 is a conservative sensitivity.")
    # Tunable constants; defaults are intentionally conservative/simple.
    ap.add_argument("--c-pre-yu", type=float, default=1.0)
    ap.add_argument("--c-pre-qfm", type=float, default=1.0)
    ap.add_argument("--c-yu-oracle-lognm", type=float, default=2.0)
    ap.add_argument("--c-qfm-popcount", type=float, default=1.0)
    args = ap.parse_args()

    args.classical_methods = [x.strip() for x in str(args.classical_methods).split(",") if x.strip()]
    results_dir = ensure_dir(args.results_dir)
    data_dir = Path(args.data_dir)
    project_dir = Path(args.project_dir)
    tools = infer_tool_paths(project_dir, args)
    qp = QuantumParams(
        gate_time_ns=args.gate_time_ns,
        qae_mode=args.qae_mode,
        yu_model=args.yu_model,
        yu_epsilon=args.yu_epsilon,
        yu_basic_oracle_depth_model=args.yu_basic_oracle_depth_model,
        yu_candidate_prep_model=args.yu_candidate_prep_model,
        yu_max_k=args.yu_max_k,
        qfm_max_k=args.qfm_max_k,
        qfm_start_k=args.qfm_start_k,
        qfm_cache_update_model=args.qfm_cache_update_model,
        qfm_preprocess_model=args.qfm_preprocess_model,
        qfm_popcount_model=args.qfm_popcount_model,
        c_pre_yu=args.c_pre_yu,
        c_pre_qfm=args.c_pre_qfm,
        c_yu_oracle_lognm=args.c_yu_oracle_lognm,
        c_qfm_popcount=args.c_qfm_popcount,
    )

    table5_datasets = [canonical_dataset(x) for x in str(args.datasets).split(",") if x.strip()]
    table12_datasets = [canonical_dataset(x) for x in str(args.table12_datasets).split(",") if x.strip()]
    candidates = parse_csv_floats(args.candidate_minsup)
    table12_data_dir = Path(args.table12_data_dir) if str(args.table12_data_dir).strip() else data_dir
    manifest = dict(args=vars(args), tools=asdict(tools), quantum_params=asdict(qp), started_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    write_json(results_dir / "run_manifest.json", manifest)

    print(f"[info] mode={args.mode}")
    print(f"[info] table5_datasets={table5_datasets}")
    print(f"[info] table12_datasets={table12_datasets}")
    print(f"[info] table5 candidate minsup={candidates}")
    print(f"[info] results_dir={results_dir}")
    print(f"[info] DATA_DIR(table5)={data_dir}")
    print(f"[info] DATA_DIR(table12)={table12_data_dir}")
    print(f"[info] SPMF_JAR={tools.spmf_jar}")
    print(f"[info] HAMM_BIN={tools.hamm_bin}")
    print(f"[info] CICLAD_BIN={tools.ciclad_bin}")

    # Preprocess datasets needed for each selected mode. Separate dirs are supported.
    table5_infos: Dict[str, DatasetInfo] = {}
    table12_infos: Dict[str, DatasetInfo] = {}

    if args.mode in {"table5", "all"}:
        for ds in table5_datasets:
            print(f"[preprocess][table5] {ds}")
            table5_infos[ds] = preprocess_dataset(ds, data_dir, results_dir, force=args.force)
            print(f"  M={table5_infos[ds].n_transactions} N={table5_infos[ds].n_items} avg_len={table5_infos[ds].avg_tx_len:.2f}")

    if args.mode in {"table12", "all"}:
        for ds in table12_datasets:
            print(f"[preprocess][table12] {ds}")
            table12_infos[ds] = preprocess_dataset(ds, table12_data_dir, results_dir, force=args.force)
            print(f"  M={table12_infos[ds].n_transactions} N={table12_infos[ds].n_items} avg_len={table12_infos[ds].avg_tx_len:.2f}")
        run_table12(table12_infos, table12_datasets, tools, qp, args, results_dir)

    if args.mode in {"table5", "all"}:
        all_candidates: List[Dict[str, Any]] = []
        selected: Dict[str, Dict[str, Any]] = {}
        # Optimized scheduling for Table 5:
        # - parallelize across datasets only, so each dataset can apply monotonic timeout pruning over minsup.
        # - within each dataset, candidate minsup values are evaluated high -> low.
        if int(args.jobs) <= 1:
            for ds in table5_datasets:
                rows, best = evaluate_dataset_optimized(ds, table5_infos[ds], candidates, tools, qp, args)
                all_candidates.extend(rows)
                if best is not None:
                    selected[ds] = best
        else:
            with ThreadPoolExecutor(max_workers=int(args.jobs)) as ex:
                futs = {ex.submit(evaluate_dataset_optimized, ds, table5_infos[ds], candidates, tools, qp, args): ds for ds in table5_datasets}
                for fut in as_completed(futs):
                    ds = futs[fut]
                    try:
                        rows, best = fut.result()
                    except Exception as e:
                        print(f"[error][{ds}] dataset evaluation failed: {type(e).__name__}: {e}", file=sys.stderr)
                        rows, best = [], None
                    all_candidates.extend(rows)
                    if best is not None:
                        selected[ds] = best

        # Write detailed JSON for reproducibility.
        write_json(results_dir / "table5_candidates_full.json", all_candidates)
        write_json(results_dir / "table5_selected_full.json", selected)

        candidate_rows = [flatten_candidate(r) for r in sorted(all_candidates, key=lambda x: (x["dataset"], -float(x["score"]))) ]
        write_csv(results_dir / "table5_candidates.csv", candidate_rows)

        selected_rows = [flatten_candidate(selected[ds]) for ds in table5_datasets if ds in selected]
        write_csv(results_dir / "table5_selected_minsup.csv", selected_rows)

        final_rows: List[Dict[str, Any]] = []
        quantum_rows: List[Dict[str, Any]] = []
        for ds in table5_datasets:
            row = selected.get(ds)
            if not row:
                continue
            final_rows.extend(final_rows_for_selected(row))
            for q in [row["qfm"], row["yu"]]:
                quantum_rows.append(dict(
                    dataset=ds,
                    method=q["method"],
                    minsup_percent=row["minsup_percent"],
                    minsup_count=row["minsup_count"],
                    n_transactions=row["n_transactions"],
                    n_items=row["n_items"],
                    preprocessing_depth=q["preprocessing_depth"],
                    body_one_query_depth=q["body_one_query_depth"],
                    total_depth=q["total_depth"],
                    runtime_sec=q["runtime_sec"],
                    max_k=q["max_k"],
                    model=q["model"],
                ))
                write_csv(results_dir / f"levels_table5_{ds}_{q['method']}.csv", q["levels"])

        write_csv(results_dir / "table5_quantum_selected_full.csv", quantum_rows)
        write_csv(results_dir / "table5_final.csv", final_rows)

        print(f"[done] wrote {results_dir / 'table5_candidates.csv'}")
        print(f"[done] wrote {results_dir / 'table5_selected_minsup.csv'}")
        print(f"[done] wrote {results_dir / 'table5_quantum_selected_full.csv'}")
        print(f"[done] wrote {results_dir / 'table5_final.csv'}")
        print("[note] For paper use, report table5_candidates.csv as the selection audit trail.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
