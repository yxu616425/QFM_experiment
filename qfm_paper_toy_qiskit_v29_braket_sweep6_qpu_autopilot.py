#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Braket sweep (6 toy points) to plot:
  x-axis: minsup / tx_count (sigma/M) ratio
  y-axis: *quantum RUNNING time* (best-effort, excludes QUEUED time by polling task.state())

You asked for exactly 6 pre-designed toy points:
  (1) Fixed dataset (M=6), vary minimum support sigma with equal spacing: sigma ∈ {1,3,5}
  (2) Fixed minimum support (absolute, non-ratio) sigma=2, vary dataset size with equal spacing: M ∈ {4,6,8}

This script is intentionally NISQ-friendly:
  - It times the core "threshold-oracle style" evaluation used in your QFc verification
    (no Grover iterations).
  - It does NOT rely on OpenQASM export. We submit Qiskit circuits through the
    qiskit-braket-provider (BraketProvider). This avoids Braket OpenQASM limitations
    such as: include statements not supported, and single-register restrictions.

Important note about timing:
  - Braket's quantum task spends time in states like QUEUED and RUNNING.
  - We measure "RUNNING-only" time as: (time when we first observe state==RUNNING) to
    (time when we first observe a terminal state).
  - This is approximate to within your polling interval.

Outputs:
  --out-csv  CSV with all 6 points and measured timings
  --out-png  scatter plot of (sigma/M) vs (RUNNING time)

Usage (SV1):
  python qfm_paper_toy_qiskit_v23_braket_sweep6_noqueue.py \
    --braket-backend SV1 --shots 2000 --poll-interval 1 \
    --out-csv sweep.csv --out-png sweep.png

Optional correctness check (adds result download and p(flag=1) estimation):
  --verify

"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# ---------- Toy point design (precomputed & asserted) ----------

ITEMS_ABC = ["A", "B", "C"]


def _as_frozenset_tx(tx: Iterable[str]) -> frozenset[str]:
    return frozenset(str(x) for x in tx)


def support_count(txs: Sequence[frozenset[str]], itemset: Sequence[str]) -> int:
    s = set(itemset)
    return sum(1 for t in txs if s.issubset(t))


def gen_C2_from_F1(F1: Sequence[str]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    F1s = sorted(F1)
    for i in range(len(F1s)):
        for j in range(i + 1, len(F1s)):
            out.append((F1s[i], F1s[j]))
    return out


@dataclass(frozen=True)
class ToyPoint:
    group: str
    name: str
    txs: List[frozenset[str]]
    sigma: int
    # expected (precomputed) results to "prove" the toy points differ
    exp_F1: List[Tuple[str]]
    exp_C2: List[Tuple[str, str]]
    exp_F2: List[Tuple[str, str]]


def build_toy_points() -> List[ToyPoint]:
    """
    Construct 6 toy points and embed expected results.

    Design rationale (your constraints):
      - Extremely small (items <=3, M<=8)
      - Distinct mining outcomes across points (avoid flat line)
      - Equal-spacing for sigma in first trio; equal-spacing for M in second trio.
    """
    pts: List[ToyPoint] = []

    # --- Group A: fixed dataset M=6, sigma in {1,3,5}
    # Dataset (M=6): 3×{A,B}, 2×{A,C}, 1×{A}
    # Supports: sup(A)=6, sup(B)=3, sup(C)=2, sup(AB)=3, sup(AC)=2, sup(BC)=0
    txs_M6_fixed = [
        _as_frozenset_tx(["A", "B"]),
        _as_frozenset_tx(["A", "B"]),
        _as_frozenset_tx(["A", "B"]),
        _as_frozenset_tx(["A"]),
        _as_frozenset_tx(["A", "C"]),
        _as_frozenset_tx(["A", "C"]),
    ]

    # sigma=1 => F1={A,B,C}, C2={AB,AC,BC}, F2={AB,AC}
    pts.append(
        ToyPoint(
            group="fixed_dataset_vary_minsup",
            name="fixedM6_sigma1",
            txs=txs_M6_fixed,
            sigma=1,
            exp_F1=[("A",), ("B",), ("C",)],
            exp_C2=[("A", "B"), ("A", "C"), ("B", "C")],
            exp_F2=[("A", "B"), ("A", "C")],
        )
    )
    # sigma=3 => F1={A,B}, C2={AB}, F2={AB}
    pts.append(
        ToyPoint(
            group="fixed_dataset_vary_minsup",
            name="fixedM6_sigma3",
            txs=txs_M6_fixed,
            sigma=3,
            exp_F1=[("A",), ("B",)],
            exp_C2=[("A", "B")],
            exp_F2=[("A", "B")],
        )
    )
    # sigma=5 => F1={A}, C2={}, F2={}
    pts.append(
        ToyPoint(
            group="fixed_dataset_vary_minsup",
            name="fixedM6_sigma5",
            txs=txs_M6_fixed,
            sigma=5,
            exp_F1=[("A",)],
            exp_C2=[],
            exp_F2=[],
        )
    )

    # --- Group B: fixed minsup (absolute) sigma=2, vary M in {4,6,8}
    # M=4: [AB, AB, A, A] => F1={A,B}, C2={AB}, F2={AB}
    txs_M4_sigma2 = [
        _as_frozenset_tx(["A", "B"]),
        _as_frozenset_tx(["A", "B"]),
        _as_frozenset_tx(["A"]),
        _as_frozenset_tx(["A"]),
    ]
    pts.append(
        ToyPoint(
            group="fixed_minsup_vary_dataset",
            name="M4_sigma2",
            txs=txs_M4_sigma2,
            sigma=2,
            exp_F1=[("A",), ("B",)],
            exp_C2=[("A", "B")],
            exp_F2=[("A", "B")],
        )
    )

    # M=6: [AB,AB, AC,AC, BC,BC] => all pairs have support=2 => F2 has 3 pairs
    txs_M6_sigma2 = [
        _as_frozenset_tx(["A", "B"]),
        _as_frozenset_tx(["A", "B"]),
        _as_frozenset_tx(["A", "C"]),
        _as_frozenset_tx(["A", "C"]),
        _as_frozenset_tx(["B", "C"]),
        _as_frozenset_tx(["B", "C"]),
    ]
    pts.append(
        ToyPoint(
            group="fixed_minsup_vary_dataset",
            name="M6_sigma2",
            txs=txs_M6_sigma2,
            sigma=2,
            exp_F1=[("A",), ("B",), ("C",)],
            exp_C2=[("A", "B"), ("A", "C"), ("B", "C")],
            exp_F2=[("A", "B"), ("A", "C"), ("B", "C")],
        )
    )

    # M=8: [AB,AB, AC,AC, A,A, B, C] => AB=2, AC=2, BC=0 => F2 has {AB,AC}
    txs_M8_sigma2 = [
        _as_frozenset_tx(["A", "B"]),
        _as_frozenset_tx(["A", "B"]),
        _as_frozenset_tx(["A", "C"]),
        _as_frozenset_tx(["A", "C"]),
        _as_frozenset_tx(["A"]),
        _as_frozenset_tx(["A"]),
        _as_frozenset_tx(["B"]),
        _as_frozenset_tx(["C"]),
    ]
    pts.append(
        ToyPoint(
            group="fixed_minsup_vary_dataset",
            name="M8_sigma2",
            txs=txs_M8_sigma2,
            sigma=2,
            exp_F1=[("A",), ("B",), ("C",)],
            exp_C2=[("A", "B"), ("A", "C"), ("B", "C")],
            exp_F2=[("A", "B"), ("A", "C")],
        )
    )

    return pts


def assert_point_precomputed(pt: ToyPoint) -> Tuple[List[str], List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Compute F1/C2/F2 using minsup_low == sigma and assert they match the precomputed expectations."""
    items = ITEMS_ABC
    txs = pt.txs
    sigma = int(pt.sigma)

    F1 = [(i,) for i in items if support_count(txs, [i]) >= sigma]
    F1_flat = [x[0] for x in F1]
    C2 = gen_C2_from_F1(F1_flat)
    F2 = [c for c in C2 if support_count(txs, c) >= sigma]

    # normalize for comparison
    F1_sorted = sorted(tuple(x) for x in F1)
    C2_sorted = sorted(tuple(x) for x in C2)
    F2_sorted = sorted(tuple(x) for x in F2)

    if F1_sorted != sorted(tuple(x) for x in pt.exp_F1):
        raise AssertionError(f"[{pt.name}] F1 mismatch: got={F1_sorted} exp={pt.exp_F1}")
    if C2_sorted != sorted(tuple(x) for x in pt.exp_C2):
        raise AssertionError(f"[{pt.name}] C2 mismatch: got={C2_sorted} exp={pt.exp_C2}")
    if F2_sorted != sorted(tuple(x) for x in pt.exp_F2):
        raise AssertionError(f"[{pt.name}] F2 mismatch: got={F2_sorted} exp={pt.exp_F2}")

    return F1_flat, C2, F2


# ---------- Qiskit circuit building (small, hardware-friendly) ----------

def _ceil_log2(x: int) -> int:
    if x <= 1:
        return 1
    return int(math.ceil(math.log2(x)))


def controlled_increment(qc, sum_qubits: List[int], ctrl: int) -> None:
    """
    Controlled increment of a little-endian integer in-place:
      sum = sum + 1  (mod 2^n)

    IMPORTANT: The gate order matters.
    We must apply carries from MSB -> LSB so that the carry condition is evaluated on the
    *pre-increment* lower bits (otherwise it breaks).
    Example (2 bits): CCX(ctrl, b0, b1) then CX(ctrl, b0).
    """
    # sum_qubits are little-endian (LSB at index 0).
    n = len(sum_qubits)
    # Apply multi-controlled flips from MSB down to LSB.
    for b in reversed(range(n)):
        controls = [ctrl] + [sum_qubits[j] for j in range(b)]
        qc.mcx(controls, sum_qubits[b])


def set_flag_if_sum_ge_sigma(qc, sum_qubits: List[int], flag: int, sigma: int, M: int) -> None:
    """
    Flip flag qubit iff sum in [sigma, M].
    sum is little-endian; we implement as OR of equality checks:
      for v in sigma..M: if sum == v then X(flag)
    This is fine for tiny M (<=8).
    """
    n = len(sum_qubits)
    for v in range(int(sigma), int(M) + 1):
        bits = [(v >> b) & 1 for b in range(n)]  # little-endian
        # map condition sum == v to all-ones controls:
        for b, bit in enumerate(bits):
            if bit == 0:
                qc.x(sum_qubits[b])
        qc.mcx(sum_qubits, flag)
        for b, bit in enumerate(bits):
            if bit == 0:
                qc.x(sum_qubits[b])


def build_threshold_oracle_circuit(
    txs: Sequence[frozenset[str]],
    items: Sequence[str],
    candidate: Tuple[str, str],
    sigma: int,
) -> Tuple["QuantumCircuit", int]:
    """
    Build a *deterministic* threshold-oracle style circuit:

      Inputs (prepared as basis state):
        - candidate bits (N=3): indicate which items are in the candidate
      Internal:
        - sum register counts how many tx contain the candidate (subset test)
        - flag flips iff sum >= sigma

    This matches the "heavy" part you are timing in QFc verification: subset checks + controlled adds + compare.
    For NISQ, you are *not* doing Grover here; you only time this oracle evaluation.

    Returns:
      (qc, flag_qubit_index)
    """
    from qiskit import QuantumCircuit, QuantumRegister

    M = len(txs)
    N = len(items)
    if N != 3:
        raise ValueError("This toy builder is specialized to 3 items (A,B,C).")

    # registers
    q_cand = QuantumRegister(N, "cand")      # candidate bitmask
    q_sum = QuantumRegister(_ceil_log2(M + 1), "sum")  # support counter (little-endian)
    q_match = QuantumRegister(1, "m")        # per-transaction match
    q_flag = QuantumRegister(1, "flag")      # threshold flag
    qc = QuantumCircuit(q_cand, q_sum, q_match, q_flag, name=f"thr_{candidate}_s{sigma}_M{M}")

    # Prepare candidate bits (basis) for this circuit
    cand_set = set(candidate)
    for i, it in enumerate(items):
        if it in cand_set:
            qc.x(q_cand[i])

    # For each transaction, compute match = AND_{i where tx lacks item i} (NOT cand[i])
    # then controlled-increment sum by match, then uncompute match.
    for tx in txs:
        missing = [i for i, it in enumerate(items) if it not in tx]
        if len(missing) == 0:
            # tx has all items => any candidate is subset; match is always 1.
            qc.x(q_match[0])
        else:
            # Use negative controls: require cand[i]==0 for all missing i.
            for i in missing:
                qc.x(q_cand[i])
            qc.mcx([q_cand[i] for i in missing], q_match[0])
            for i in missing:
                qc.x(q_cand[i])

        # increment sum if match == 1
        controlled_increment(qc, [qb for qb in q_sum], q_match[0])

        # uncompute match to |0>
        if len(missing) == 0:
            qc.x(q_match[0])
        else:
            for i in missing:
                qc.x(q_cand[i])
            qc.mcx([q_cand[i] for i in missing], q_match[0])
            for i in missing:
                qc.x(q_cand[i])

    # Compare sum >= sigma and flip flag
    set_flag_if_sum_ge_sigma(qc, [qb for qb in q_sum], q_flag[0], sigma=int(sigma), M=M)

    return qc, qc.qubits.index(q_flag[0])


def attach_measure_all_with_flag_to_c0(qc: "QuantumCircuit", flag_qubit_index: int) -> Tuple["QuantumCircuit", int]:
    """
    Braket on-demand simulators (SV1/TN1) require all qubits to be measured in the program.

    We measure *all* qubits, but we permute the measurement mapping so that:
      classical bit c[0] measures the flag qubit.

    Returns:
      (qc_meas, flag_cbit_index==0)
    """
    from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister

    n = qc.num_qubits
    q = QuantumRegister(n, "q")
    c = ClassicalRegister(n, "c")

    out = QuantumCircuit(q, c, name=qc.name + "_meas")

    # Compose by POSITION (works even if `qc` uses multiple registers).
    out.compose(qc, qubits=out.qubits, inplace=True)

    # measure flag -> c[0]
    out.measure(out.qubits[flag_qubit_index], c[0])

    # measure remaining qubits -> c[1..]
    ci = 1
    for qi in range(n):
        if qi == flag_qubit_index:
            continue
        out.measure(out.qubits[qi], c[ci])
        ci += 1

    return out, 0



def find_qubit_measured_to_c0(qc_meas: "QuantumCircuit") -> int:
    """
    Return the *qubit index* that is measured into classical bit c[0].

    IMPORTANT: do NOT rely on object identity for Clbit, because some Qiskit versions/providers
    may copy objects during circuit building/compose. We match by classical-bit index instead.
    """
    # Determine which clbit is "c0" by index.
    target_c_index = 0

    for ci in qc_meas.data:
        # CircuitInstruction (new) vs tuple (legacy). Avoid treating CircuitInstruction as iterable.
        if hasattr(ci, "operation"):
            inst = ci.operation
            qargs = ci.qubits
            cargs = ci.clbits
        else:
            inst, qargs, cargs = ci

        if getattr(inst, "name", "") != "measure":
            continue
        if not cargs:
            continue

        # Map the measured clbit to its index in qc_meas
        try:
            c_index = qc_meas.find_bit(cargs[0]).index  # type: ignore[attr-defined]
        except Exception:
            # fallback: linear search
            try:
                c_index = list(qc_meas.clbits).index(cargs[0])
            except Exception:
                continue

        if c_index == target_c_index:
            # Return the qubit index of the measured qubit
            try:
                return qc_meas.find_bit(qargs[0]).index  # type: ignore[attr-defined]
            except Exception:
                return list(qc_meas.qubits).index(qargs[0])

    raise RuntimeError("Could not find which qubit is measured to c[0] in qc_meas.")

def infer_c0_leftmost_from_counts(counts: Dict[str, int]) -> bool:
    """
    Detect whether the returned counts key string places c0 at the LEFTMOST character.
    We assume the circuit prepared c0 deterministically as 1.
    """
    if not counts:
        return False
    key = max(counts.items(), key=lambda kv: kv[1])[0]
    key = str(key).replace(" ", "")
    if not key:
        return False
    if key[0] == "1":
        return True
    if key[-1] == "1":
        return False
    return False


# ---------- Braket execution and "no-queue" timing ----------

TERMINAL_STATES = {"COMPLETED", "FAILED", "CANCELLED", "CANCELLING"}


# ---------- QPU helper: region, polling, pricing, sanity ----------

# Prices are from the AWS Braket public pricing page. Update if AWS changes pricing.
# Format: family -> (per_task_usd, per_shot_usd)
QPU_PRICING_USD: Dict[str, Tuple[float, float]] = {
    "AQT IBEX-Q1": (0.30, 0.02350),
    "IONQ ARIA": (0.30, 0.03000),
    "IONQ FORTE": (0.30, 0.08000),
    "IQM EMERALD": (0.30, 0.00160),
    "IQM GARNET": (0.30, 0.00145),
    "QUERA AQUILA": (0.30, 0.01000),
    "RIGETTI ANKAA": (0.30, 0.00090),
}


def _median(xs: Sequence[float]) -> float:
    ys = sorted(float(x) for x in xs)
    if not ys:
        return float("nan")
    n = len(ys)
    mid = n // 2
    if n % 2 == 1:
        return ys[mid]
    return 0.5 * (ys[mid - 1] + ys[mid])


def _is_qpu_backend(backend_name: str) -> bool:
    bn = str(backend_name).strip().upper()
    if bn in {"SV1", "TN1", "DM1"}:
        return False
    return True


def _normalize_qpu_family(backend_name: str) -> Optional[str]:
    """Best-effort mapping from backend name/ARN to a pricing family key."""
    bn = str(backend_name).strip().upper()

    # ARN form: arn:aws:braket:region::device/qpu/<provider>/<device>
    if "ARN:AWS:BRAKET" in bn:
        if "/IONQ/" in bn and "FORTE" in bn:
            return "IONQ FORTE"
        if "/IONQ/" in bn and "ARIA" in bn:
            return "IONQ ARIA"
        if "/RIGETTI/" in bn and "ANKAA" in bn:
            return "RIGETTI ANKAA"
        if "/IQM/" in bn and "EMERALD" in bn:
            return "IQM EMERALD"
        if "/IQM/" in bn and "GARNET" in bn:
            return "IQM GARNET"
        if "/AQT/" in bn and "IBEX" in bn:
            return "AQT IBEX-Q1"
        if "/QUERA/" in bn and "AQUILA" in bn:
            return "QUERA AQUILA"

    # Name form
    if "FORTE" in bn:
        return "IONQ FORTE"
    if "ARIA" in bn:
        return "IONQ ARIA"
    if "ANKAA" in bn:
        return "RIGETTI ANKAA"
    if "EMERALD" in bn:
        return "IQM EMERALD"
    if "GARNET" in bn:
        return "IQM GARNET"
    if "IBEX" in bn:
        return "AQT IBEX-Q1"
    if "AQUILA" in bn:
        return "QUERA AQUILA"

    return None


def _apply_aws_region(region: Optional[str]) -> None:
    """Force AWS region for boto3-based backends (best-effort)."""
    if not region:
        return
    os.environ["AWS_DEFAULT_REGION"] = str(region)
    os.environ["AWS_REGION"] = str(region)
    try:
        import boto3  # type: ignore
        boto3.setup_default_session(region_name=str(region))
    except Exception:
        pass


def _enforce_backend_shot_rules(backend_name: str, shots: int) -> None:
    bn = str(backend_name).strip().upper()
    # IonQ requires minimum 100 shots per task.
    if ("IONQ" in bn) or ("ARIA" in bn) or ("FORTE" in bn):
        if int(shots) < 100:
            raise ValueError(f"IonQ QPUs require shots >= 100 per task (got {shots}).")
    # QuEra Aquila has a max shots of 1000.
    if "AQUILA" in bn:
        if int(shots) > 1000:
            raise ValueError(f"QuEra Aquila max shots is 1000 per task (got {shots}).")


def _estimate_qpu_cost_usd(
    backend_name: str,
    num_tasks: int,
    shots_per_task: int,
    extra_tasks: int = 0,
    extra_shots: int = 0,
    price_per_task: Optional[float] = None,
    price_per_shot: Optional[float] = None,
) -> Tuple[Optional[float], str]:
    """
    Estimate QPU on-demand cost in USD. Returns (cost_or_None, meta_string).
    If pricing is unknown and no overrides are provided, returns (None, ...).
    """
    fam = _normalize_qpu_family(backend_name)

    # If user didn't override, try built-in.
    if price_per_task is None and price_per_shot is None and fam in QPU_PRICING_USD:
        price_per_task, price_per_shot = QPU_PRICING_USD[fam]

    if price_per_task is None or price_per_shot is None:
        return None, f"pricing_unknown(family={fam})"

    total_tasks = int(num_tasks) + int(extra_tasks)
    total_shots = int(num_tasks) * int(shots_per_task) + int(extra_shots)
    cost = total_tasks * float(price_per_task) + total_shots * float(price_per_shot)
    meta = f"family={fam} per_task={price_per_task} per_shot={price_per_shot}"
    return float(cost), meta


def _filter_points(pts: List["ToyPoint"], spec: Optional[str]) -> Tuple[List["ToyPoint"], List[str]]:
    """
    Filter toy points by comma-separated names or groups.
    Returns (filtered_pts, missing_tokens).
    """
    if not spec:
        return pts, []
    tokens = [t.strip() for t in str(spec).split(",") if t.strip()]
    if not tokens:
        return pts, []
    token_set = set(tokens)

    names = {p.name for p in pts}
    groups = {p.group for p in pts}

    filtered: List["ToyPoint"] = []
    for p in pts:
        if (p.name in token_set) or (p.group in token_set):
            filtered.append(p)

    missing = sorted([t for t in token_set if (t not in names and t not in groups)])
    return filtered, missing


def _planned_circuits_for_points(pts: List["ToyPoint"], repeats: int) -> int:
    total = 0
    for p in pts:
        _, C2, _ = assert_point_precomputed(p)
        total += len(C2) * int(repeats)
    return total


def _get_tasks_from_job(job) -> List[object]:
    """Best-effort extraction of AwsQuantumTask objects from a qiskit-braket-provider Job."""
    for attr in ("tasks", "_tasks"):
        ts = getattr(job, attr, None)
        if ts:
            try:
                return list(ts)
            except Exception:
                pass
    return []


def run_circuit_braket_noqueue_time(
    backend_name: str,
    qc_meas: "QuantumCircuit",
    shots: int,
    poll_interval: float,
    poll_timeout: float,
    verify: bool,
    c0_leftmost_cache: Dict[str, Optional[bool]],
) -> Tuple[float, float, float, str, Optional[float]]:
    """
    Submit ONE circuit to Braket backend (via qiskit-braket-provider),
    and measure "RUNNING-only" time by polling task.state().

    Returns:
      (running_time_s, total_time_s, queue_est_s, final_state, p_flag1_if_verify_else_None)

    Notes:
      - running_time is approximated to within poll_interval.
      - total_time includes queue + running + client overhead.
      - queue_est = total_time - running_time (best-effort).
      - p_flag1 is estimated from measurement results. We assume the circuit measures the flag qubit into c[0],
        but the returned bitstring may place c0 leftmost or rightmost depending on provider; we calibrate once.
    """
    from qiskit_braket_provider import BraketProvider

    provider = BraketProvider()
    backend = provider.get_backend(backend_name)

    t_submit = time.time()
    job = backend.run(qc_meas, shots=int(shots))

    tasks = _get_tasks_from_job(job)
    if not tasks:
        # Fallback: cannot separate queue time; use wall time around job.result()
        p_flag1 = None
        if verify:
            res = job.result()
            try:
                counts = res.get_counts()
            except Exception:
                counts = res.get_counts(0)
            # Use default qiskit convention: c0 is rightmost.
            ones = sum(v for k, v in counts.items() if str(k).replace(" ", "")[-1] == "1")
            tot = sum(int(v) for v in counts.values())
            p_flag1 = (ones / tot) if tot else None
        t_end = time.time()
        return float("nan"), t_end - t_submit, float("nan"), "UNKNOWN_NO_TASKS", p_flag1

    # Poll all tasks (for one circuit it should be exactly 1 task).
    t_running_first: Optional[float] = None
    t_terminal_last: Optional[float] = None
    final_states: List[str] = []

    for task in tasks:
        t0 = time.time()
        t_run: Optional[float] = None
        state = "UNKNOWN"
        while True:
            if time.time() - t0 > float(poll_timeout):
                state = "TIMEOUT"
                break
            try:
                state = str(task.state()).upper()
            except Exception:
                state = "STATE_ERR"
                break

            if state == "RUNNING" and t_run is None:
                t_run = time.time()
            if state in TERMINAL_STATES:
                break
            time.sleep(float(poll_interval))

        t_term = time.time()
        final_states.append(state)
        if t_run is None:
            # We never observed RUNNING; best-effort: treat as 0 running from submit.
            t_run = t_submit
        if t_running_first is None or t_run < t_running_first:
            t_running_first = t_run
        if t_terminal_last is None or t_term > t_terminal_last:
            t_terminal_last = t_term

    t_end_total = time.time()
    running_time = float(t_terminal_last - t_running_first) if (t_running_first and t_terminal_last) else float("nan")
    total_time = t_end_total - t_submit
    queue_est = total_time - running_time if (not math.isnan(running_time)) else float("nan")

    # Optional correctness estimation (downloads results; not included in running_time)
    p_flag1: Optional[float] = None
    if verify:
        # Download results (NOT counted in running_time)
        res = job.result()
        try:
            counts = res.get_counts()
        except Exception:
            counts = res.get_counts(0)

        # Calibrate once per backend: does the bitstring place c0 on the leftmost or rightmost?
        cache_key = backend_name
        c0_leftmost = c0_leftmost_cache.get(cache_key, None)
        if c0_leftmost is None:
            from qiskit import QuantumCircuit
            q_to_c0 = find_qubit_measured_to_c0(qc_meas)
            cal = QuantumCircuit(qc_meas.num_qubits, qc_meas.num_clbits)
            cal.x(q_to_c0)
            # measure ALL qubits with the same "q_to_c0 -> c0" mapping
            cal.measure(q_to_c0, 0)
            ci = 1
            for q in range(cal.num_qubits):
                if q == q_to_c0:
                    continue
                cal.measure(q, ci)
                ci += 1

            cal_job = backend.run(cal, shots=min(256, int(shots)))
            cal_res = cal_job.result()
            try:
                cal_counts = cal_res.get_counts()
            except Exception:
                cal_counts = cal_res.get_counts(0)

            c0_leftmost = infer_c0_leftmost_from_counts(cal_counts)
            c0_leftmost_cache[cache_key] = c0_leftmost

        ones = 0
        tot = 0
        for k, v in counts.items():
            ks = str(k).replace(" ", "")
            if not ks:
                continue
            bit = ks[0] if c0_leftmost else ks[-1]
            ones += int(bit == "1") * int(v)
            tot += int(v)
        if tot > 0:
            p_flag1 = ones / tot

    final_state = final_states[-1] if final_states else "UNKNOWN"
    return running_time, total_time, queue_est, final_state, p_flag1


# ---------- Main sweep ----------

@dataclass
class SweepRow:
    group: str
    name: str
    M: int
    sigma: int
    ratio: float
    C2_size: int
    F2_size: int
    repeats: int
    running_s: float
    running_s_median: float
    running_s_mean: float
    running_s_p25: float
    running_s_p75: float
    running_s_list_json: str
    total_s: float
    queue_est_s: float
    status: str
    checked: int
    mismatches: int
    pass_verify: str


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--braket-backend", type=str, default="SV1",
                    help="Braket backend name, e.g., SV1, TN1, dm1, Ankaa-3, Forte 1 ...")

    ap.add_argument("--aws-region", type=str, default=None,
                    help="Force AWS region (sets AWS_DEFAULT_REGION/AWS_REGION). Useful when targeting QPUs in another region.")
    ap.add_argument("--device-arn", type=str, default=None,
                    help="Optional Braket device ARN. If set, overrides --braket-backend.")
    ap.add_argument("--points", type=str, default=None,
                    help="Comma-separated list of toy point names or groups to run (e.g., 'M4_sigma2,M6_sigma2' or 'fixed_minsup_vary_dataset').")
    ap.add_argument("--sanity", action="store_true",
                    help="Run a low-cost sanity check first (recommended before QPU runs).")
    ap.add_argument("--sanity-only", action="store_true",
                    help="Run sanity check and exit without running the full sweep.")
    ap.add_argument("--sanity-point", type=str, default="M4_sigma2",
                    help="Toy point name used for sanity check (default: M4_sigma2).")
    ap.add_argument("--sanity-repeats", type=int, default=1,
                    help="Number of repeats in sanity check (default: 1).")
    ap.add_argument("--sanity-max-cands", type=int, default=1,
                    help="Max number of candidates (circuits) to run in sanity check (default: 1).")
    ap.add_argument("--estimate-only", action="store_true",
                    help="Print estimated number of QPU tasks/shots/cost and exit.")
    ap.add_argument("--max-estimated-cost-usd", type=float, default=None,
                    help="Abort if the estimated QPU cost exceeds this value (USD).")
    ap.add_argument("--c0-bit-order", type=str, default="auto", choices=["auto", "left", "right"],
                    help="How to interpret which side of the returned bitstring is classical bit c0. "
                         "'auto' submits one calibration circuit per backend when --verify is enabled; "
                         "'left'/'right' avoids calibration on QPU.")
    ap.add_argument("--auto-poll-interval-qpu", action="store_true",
                    help="If set and backend is QPU, increase --poll-interval to at least --poll-interval-qpu.")
    ap.add_argument("--poll-interval-qpu", type=float, default=5.0,
                    help="Suggested polling interval for QPUs when --auto-poll-interval-qpu is set.")
    ap.add_argument("--qpu-price-per-task", type=float, default=None,
                    help="Override QPU per-task price (USD). If set, also set --qpu-price-per-shot.")
    ap.add_argument("--qpu-price-per-shot", type=float, default=None,
                    help="Override QPU per-shot price (USD). If set, also set --qpu-price-per-task.")
    ap.add_argument("--print-plan", action="store_true",
                    help="Print recommended example commands (SV1 sanity + IonQ Forte) and exit.")
    ap.add_argument("--shots", type=int, default=2000)
    ap.add_argument("--poll-interval", type=float, default=1.0)
    ap.add_argument("--poll-timeout", type=float, default=3600.0)
    ap.add_argument("--out-csv", type=str, default="sweep.csv")

    # Backward compatible: if you pass --out-png sweep.png, we will write two figures:
    #   sweep_ms.png and sweep_tx.png
    ap.add_argument("--out-png", type=str, default=None,
                    help="(compat) Base png name. Will emit <base>_ms.png and <base>_tx.png. "
                         "If omitted, use --out-png-ms/--out-png-tx.")
    ap.add_argument("--out-png-ms", type=str, default="sweep_ms.png",
                    help="Output png for x-axis=ms (sigma) (first 3 points).")
    ap.add_argument("--out-png-tx", type=str, default="sweep_tx.png",
                    help="Output png for x-axis=tx count M (last 3 points).")

    ap.add_argument("--verify", action="store_true",
                    help="Check quantum measurement correctness: estimate p(flag=1) and compare with (support>=sigma).")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Decision threshold on p(flag=1) for verify (>=threshold means True).")
    ap.add_argument("--repeats", type=int, default=5,
                    help="Repeat each toy point N times; we aggregate runtime excluding queue.")
    ap.add_argument("--agg", type=str, default="median", choices=["median", "mean"],
                    help="Aggregation for plotting/CSV running_s over repeats.")
    ap.add_argument("--verify-verbose", action="store_true",
                    help="If set, print per-candidate verify lines for every repeat. Otherwise prints only for the first repeat and mismatches.")

    args = ap.parse_args()

    # --- QPU/QS settings ---
    _apply_aws_region(args.aws_region)

    backend_id = (str(args.device_arn).strip() if args.device_arn else str(args.braket_backend).strip())

    if args.print_plan:
        print("\n[Plan] Example commands")
        print("  # 1) Low-cost sanity run on SV1 (recommended first)")
        print("  python {} --braket-backend SV1 --shots 100 --repeats 1 --agg median --poll-interval 0.2 --verify --sanity --sanity-only".format(os.path.basename(__file__)))
        print("")
        print("  # 2) Estimate cost on IonQ Forte (on-demand) — use the correct region/device name or ARN")
        print("  python {} --braket-backend 'Forte 1' --aws-region us-east-1 --shots 100 --repeats 1 --agg median --poll-interval 5 --auto-poll-interval-qpu --verify --c0-bit-order right --estimate-only".format(os.path.basename(__file__)))
        print("")
        print("  # 3) Actually run on QPU (be careful with cost)")
        print("  python {} --braket-backend 'Forte 1' --aws-region us-east-1 --shots 100 --repeats 1 --agg median --poll-interval 5 --auto-poll-interval-qpu --verify --c0-bit-order right --max-estimated-cost-usd 200".format(os.path.basename(__file__)))
        return

    # Auto-relax polling interval for QPUs when requested (or in sanity mode)
    if (args.auto_poll_interval_qpu or args.sanity or args.sanity_only) and _is_qpu_backend(backend_id):
        args.poll_interval = max(float(args.poll_interval), float(args.poll_interval_qpu))

    # Enforce per-device shot constraints (best-effort)
    _enforce_backend_shot_rules(backend_id, int(args.shots))

    # Resolve output filenames
    out_png_ms = args.out_png_ms
    out_png_tx = args.out_png_tx
    if args.out_png:
        base = str(args.out_png)
        if base.lower().endswith(".png"):
            stem = base[:-4]
            out_png_ms = stem + "_ms.png"
            out_png_tx = stem + "_tx.png"
        else:
            out_png_ms = base + "_ms.png"
            out_png_tx = base + "_tx.png"

    pts_all = build_toy_points()
    pts, missing = _filter_points(pts_all, args.points)
    if missing:
        print(f"[Warn] Unknown point/group tokens in --points: {missing}")
    if not pts:
        raise ValueError("No toy points selected (check --points).")

    print(f"[Braket] backend: {backend_id}")

    print("\n[Precomputed toy points]")
    for pt in pts:
        F1_flat, C2, F2 = assert_point_precomputed(pt)
        ratio = pt.sigma / len(pt.txs)
        print(f"  - {pt.group:24s} | {pt.name:14s} | M={len(pt.txs)} sigma={pt.sigma} ratio={ratio:.3f} | "
              f"|F1|={len(F1_flat)} |C2|={len(C2)} |F2|={len(F2)} | F2={sorted(F2)}")

    rows: List[SweepRow] = []
    c0_leftmost_cache: Dict[str, Optional[bool]] = {}

    # Apply user-specified c0 bit order (avoids calibration circuit when --verify is on)
    if str(args.c0_bit_order).lower() in {"left", "right"}:
        c0_leftmost_cache[backend_id] = (str(args.c0_bit_order).lower() == "left")

    # --- Plan / estimate ---
    planned_circuits = _planned_circuits_for_points(pts, int(args.repeats))
    extra_tasks = 0
    extra_shots = 0
    if bool(args.verify) and str(args.c0_bit_order).lower() == "auto":
        extra_tasks = 1
        extra_shots = min(256, int(args.shots))

    est_cost = None
    est_meta = "n/a"
    if _is_qpu_backend(backend_id):
        est_cost, est_meta = _estimate_qpu_cost_usd(
            backend_name=backend_id,
            num_tasks=planned_circuits,
            shots_per_task=int(args.shots),
            extra_tasks=extra_tasks,
            extra_shots=extra_shots,
            price_per_task=args.qpu_price_per_task,
            price_per_shot=args.qpu_price_per_shot,
        )

    print(f"\n[Plan] selected_points={len(pts)} circuits={planned_circuits} (repeats={args.repeats}, shots={args.shots})")
    if bool(args.verify) and str(args.c0_bit_order).lower() == "auto":
        print(f"[Plan] verify calibration: +{extra_tasks} task(s), +{extra_shots} shots (first backend only)")
    if _is_qpu_backend(backend_id):
        if est_cost is None:
            print(f"[Plan] estimated QPU cost: unknown ({est_meta}). You can provide --qpu-price-per-task/--qpu-price-per-shot.")
        else:
            print(f"[Plan] estimated QPU cost (on-demand): ${est_cost:.2f} ({est_meta})")
            if args.max_estimated_cost_usd is not None and float(est_cost) > float(args.max_estimated_cost_usd):
                raise SystemExit(f"Estimated cost ${est_cost:.2f} exceeds --max-estimated-cost-usd {args.max_estimated_cost_usd}. Aborting.")
    if args.estimate_only:
        return

    # --- Sanity check (runs BEFORE the full sweep) ---
    if args.sanity or args.sanity_only:
        sanity_pt: Optional[ToyPoint] = None
        for p in pts_all:
            if p.name == args.sanity_point:
                sanity_pt = p
                break
        if sanity_pt is None:
            for p in pts:
                _, C2_tmp, _ = assert_point_precomputed(p)
                if len(C2_tmp) > 0:
                    sanity_pt = p
                    break
        if sanity_pt is None:
            sanity_pt = pts[0]

        _, sanity_C2, _ = assert_point_precomputed(sanity_pt)
        sanity_C2_sorted = sorted(sanity_C2)
        sanity_C2_sorted = sanity_C2_sorted[:max(1, int(args.sanity_max_cands))]
        sanity_repeats = max(1, int(args.sanity_repeats))

        print(f"\n[Sanity] point={sanity_pt.name} group={sanity_pt.group} M={len(sanity_pt.txs)} sigma={int(sanity_pt.sigma)} "
              f"|C2|={len(sanity_C2)} run_cands={len(sanity_C2_sorted)} repeats={sanity_repeats}")

        sanity_wall_times: List[float] = []

        for rep in range(sanity_repeats):
            for cand in sanity_C2_sorted:
                t0 = time.time()
                qc, flag_q = build_threshold_oracle_circuit(sanity_pt.txs, ITEMS_ABC, candidate=tuple(cand), sigma=int(sanity_pt.sigma))
                qc_meas, _ = attach_measure_all_with_flag_to_c0(qc, flag_qubit_index=flag_q)

                _run_s, _total_s, _queue_s, _status, _p1 = run_circuit_braket_noqueue_time(
                    backend_name=backend_id,
                    qc_meas=qc_meas,
                    shots=int(args.shots),
                    poll_interval=float(args.poll_interval),
                    poll_timeout=float(args.poll_timeout),
                    verify=True,
                    c0_leftmost_cache=c0_leftmost_cache,
                )
                t1 = time.time()
                sanity_wall_times.append(t1 - t0)

        c0 = c0_leftmost_cache.get(backend_id, None)
        if c0 is None:
            print("[Sanity] c0_bit_order: unknown")
        else:
            print(f"[Sanity] c0_bit_order (inferred/forced): {'left' if c0 else 'right'}")

        if sanity_wall_times:
            w_med = _median(sanity_wall_times)
            print(f"[Sanity] wall_time_per_circuit_median_s={w_med:.3f} (n={len(sanity_wall_times)})")
            extra_wall_circuits = 1 if (bool(args.verify) and str(args.c0_bit_order).lower() == "auto") else 0
            est_wall_s = (planned_circuits + extra_wall_circuits) * w_med
            print(f"[Sanity] rough_wall_time_full_run_s≈{est_wall_s:.1f} (~{est_wall_s/60:.1f} min)")

        if args.sanity_only:
            return

    global_checked = 0
    global_mismatches = 0

    for pt in pts:
        _, C2, F2 = assert_point_precomputed(pt)
        M = len(pt.txs)
        sigma = int(pt.sigma)
        ratio = sigma / M

        if len(C2) == 0:
            rows.append(
                SweepRow(
                    group=pt.group,
                    name=pt.name,
                    M=M,
                    sigma=sigma,
                    ratio=ratio,
                    C2_size=0,
                    F2_size=len(F2),
                    repeats=0,
                    running_s=0.0,
                    running_s_median=0.0,
                    running_s_mean=0.0,
                    running_s_p25=0.0,
                    running_s_p75=0.0,
                    running_s_list_json="[]",
                    total_s=0.0,
                    queue_est_s=0.0,
                    status="SKIP_NO_CANDIDATES",
                    checked=0,
                    mismatches=0,
                    pass_verify="NA",
                )
            )
            continue

        print(f"\n[Run] {pt.name}: M={M} sigma={sigma} |C2|={len(C2)}")

        repeats = max(1, int(args.repeats))
        rep_running: List[float] = []
        rep_total: List[float] = []
        rep_queue: List[float] = []
        status_seen: List[str] = []

        checked_total = 0
        mismatches_total = 0

        for rep in range(repeats):
            running_sum = 0.0
            total_sum = 0.0
            queue_sum = 0.0
            status_all: List[str] = []
            checked = 0
            mismatches = 0

            for cand in C2:
                qc, flag_q = build_threshold_oracle_circuit(pt.txs, ITEMS_ABC, candidate=tuple(cand), sigma=sigma)
                qc_meas, _ = attach_measure_all_with_flag_to_c0(qc, flag_qubit_index=flag_q)

                run_s, total_s, queue_s, status, p1 = run_circuit_braket_noqueue_time(
                    backend_name=backend_id,
                    qc_meas=qc_meas,
                    shots=int(args.shots),
                    poll_interval=float(args.poll_interval),
                    poll_timeout=float(args.poll_timeout),
                    verify=bool(args.verify),
                    c0_leftmost_cache=c0_leftmost_cache,
                )

                status_all.append(status)
                if not math.isnan(run_s):
                    running_sum += float(run_s)
                if not math.isnan(total_s):
                    total_sum += float(total_s)
                if not math.isnan(queue_s):
                    queue_sum += float(queue_s)

                if args.verify:
                    sup = support_count(pt.txs, cand)
                    expect = (sup >= sigma)
                    if p1 is None or math.isnan(float(p1)):
                        got = None
                        ok = False
                    else:
                        got = (float(p1) >= float(args.threshold))
                        ok = (got == expect)

                    checked += 1
                    checked_total += 1
                    global_checked += 1
                    if not ok:
                        mismatches += 1
                        mismatches_total += 1
                        global_mismatches += 1

                    if rep == 0 or args.verify_verbose or (not ok):
                        print(f"  cand={cand} sup={sup} expect={expect}  p(flag=1)={p1} -> got={got}  status={status}")

            rep_running.append(float(running_sum))
            rep_total.append(float(total_sum))
            rep_queue.append(float(queue_sum))
            status_summary = "COMPLETED" if all(s == "COMPLETED" for s in status_all) else ";".join(status_all)
            status_seen.append(status_summary)

            if args.verify:
                pass_rep = "PASS" if mismatches == 0 else "FAIL"
                print(f"  [Verify] {pt.name} (rep {rep+1}/{repeats}): {pass_rep}  (mismatches={mismatches}/{checked}, threshold={args.threshold})")

        def _quantile(sorted_list: List[float], q: float) -> float:
            if not sorted_list:
                return 0.0
            if len(sorted_list) == 1:
                return float(sorted_list[0])
            pos = (len(sorted_list) - 1) * q
            lo = int(pos)
            hi = min(lo + 1, len(sorted_list) - 1)
            frac = pos - lo
            return float(sorted_list[lo] * (1 - frac) + sorted_list[hi] * frac)

        sorted_r = sorted(rep_running)
        running_p25 = _quantile(sorted_r, 0.25)
        running_p75 = _quantile(sorted_r, 0.75)
        running_median = _quantile(sorted_r, 0.50)
        running_mean = float(sum(rep_running) / len(rep_running)) if rep_running else 0.0

        running_agg = running_mean if args.agg == "mean" else running_median
        total_agg = float(sum(rep_total) / len(rep_total)) if rep_total else 0.0
        queue_agg = float(sum(rep_queue) / len(rep_queue)) if rep_queue else 0.0

        status = "|".join(status_seen)

        pass_verify = "NA"
        if args.verify:
            pass_verify = "PASS" if mismatches_total == 0 else "FAIL"
            print(f"  [Verify] {pt.name}: {pass_verify}  (mismatches={mismatches_total}/{checked_total}, threshold={args.threshold})")

        rows.append(
            SweepRow(
                group=pt.group,
                name=pt.name,
                M=M,
                sigma=sigma,
                ratio=ratio,
                C2_size=len(C2),
                F2_size=len(F2),
                repeats=repeats,
                running_s=running_agg,
                running_s_median=running_median,
                running_s_mean=running_mean,
                running_s_p25=running_p25,
                running_s_p75=running_p75,
                running_s_list_json=json.dumps(rep_running),
                total_s=total_agg,
                queue_est_s=queue_agg,
                status=status,
                checked=checked_total,
                mismatches=mismatches_total,
                pass_verify=pass_verify,
            )
        )

    if args.verify:
        overall = "PASS" if global_mismatches == 0 else "FAIL"
        print(f"\n[Verify Summary] overall={overall} mismatches={global_mismatches}/{global_checked} (threshold={args.threshold})")

    # Write CSV
    import csv
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "group", "name", "M", "sigma", "ratio_sigma_over_M",
            "C2_size", "F2_size",
            "running_time_s_excl_queue",
            "total_time_s", "queue_est_s",
            "status",
            "verify_checked", "verify_mismatches", "verify_pass",
        ])
        for r in rows:
            w.writerow([
                r.group, r.name, r.M, r.sigma, f"{r.ratio:.6f}",
                r.C2_size, r.F2_size,
                f"{r.running_s:.6f}",
                f"{r.total_s:.6f}",
                f"{r.queue_est_s:.6f}",
                r.status,
                r.checked, r.mismatches, r.pass_verify,
            ])

    # Plot: two figures
    import matplotlib.pyplot as plt

    # (1) x-axis = ms (sigma) for the first group (fixed dataset vary minsup)
    ms_rows = [r for r in rows if r.group == "fixed_dataset_vary_minsup"]
    ms_rows = sorted(ms_rows, key=lambda r: r.sigma)
    plt.figure()
    xs = [r.sigma for r in ms_rows]
    ys = [r.running_s for r in ms_rows]
    plt.scatter(xs, ys)
    for r in ms_rows:
        plt.annotate(r.name, (r.sigma, r.running_s))
    plt.xlabel("ms (minimum support = sigma)")
    plt.ylabel("quantum runtime (RUNNING time, seconds)  [excludes queue, best-effort]")
    plt.title(f"Braket {args.braket_backend}: fixed dataset (M=6), vary ms")
    plt.tight_layout()
    plt.savefig(out_png_ms, dpi=200)

    # (2) x-axis = tx_count (M) for the second group (fixed minsup vary dataset size)
    tx_rows = [r for r in rows if r.group == "fixed_minsup_vary_dataset"]
    tx_rows = sorted(tx_rows, key=lambda r: r.M)
    plt.figure()
    xs2 = [r.M for r in tx_rows]
    ys2 = [r.running_s for r in tx_rows]
    plt.scatter(xs2, ys2)
    for r in tx_rows:
        plt.annotate(r.name, (r.M, r.running_s))
    plt.xlabel("tx count (M)")
    plt.ylabel("quantum runtime (RUNNING time, seconds)  [excludes queue, best-effort]")
    plt.title(f"Braket {args.braket_backend}: fixed ms (sigma=2), vary tx count")
    plt.tight_layout()
    plt.savefig(out_png_tx, dpi=200)

    print(f"\n[Done] wrote CSV: {args.out_csv}")
    print(f"[Done] wrote PNG (ms): {out_png_ms}")
    print(f"[Done] wrote PNG (tx): {out_png_tx}")


if __name__ == "__main__":
    main()