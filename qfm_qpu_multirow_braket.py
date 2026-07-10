#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QFM Table-IX multi-row QPU validation on Amazon Braket (one-click).
===================================================================
Answers Yang 0701 comment [急]: "Table IX 只有一 row 太 weak，請再跑 2 組其他
sample dataset，結果類似即可" + (optional) second backend echoing the Aer
noise table (Rigetti Ankaa-class row).

What it does
------------
For each configured GROUP, it *actually samples* M transactions from a real
UCI dataset file (so the paper sentence "randomly sampled from the X dataset"
is literally true), projects them onto 3 items (attribute=value items, chosen
near the support threshold so labels are non-trivial), enumerates the C2
candidate pairs exactly like the original 11-instance Mushroom run (F1 ->
prefix join -> C2), builds the SAME v29 threshold-oracle circuit used for the
existing IonQ Forte 1 row, and runs each candidate-verification instance on
the selected Braket QPU with --shots (default 100, same protocol as the
existing row).

Default plan (matches the LINE discussion):
  Row 2: Tic-Tac-Toe, M in {5,6},  ~11 instances, IonQ Forte 1
  Row 3: Car,         M in {7,8},  ~11 instances, IonQ Forte 1
  Row 4 (optional, --with-ankaa): Chess (kr-vs-kp), M in {4,5}, ~10 instances,
         Rigetti Ankaa-3  -> echoes the "Rigetti Ankaa-class" row of the Aer
         table; expected ~90%+ (NOT guaranteed 100%: superconducting noise +
         routing; that is the point of the echo row).

Safety: DEFAULT IS DRY-RUN. It builds every circuit, verifies every label
classically, prints depths and a cost estimate, and writes the plan CSV.
Nothing is submitted until you re-run with --submit.

Usage (on SageMaker, next to the v29 file, with data_raw/ present)
------------------------------------------------------------------
  # 1) dry run (no AWS calls except optional device lookup):
  python qfm_qpu_multirow_braket.py

  # 2) real run, two IonQ rows:
  python qfm_qpu_multirow_braket.py --submit

  # 3) real run including the Ankaa echo row:
  python qfm_qpu_multirow_braket.py --submit --with-ankaa

  # useful:
  --data-dir PATH      folder containing tic-tac-toe.data / car.data / kr-vs-kp.data
  --v29 PATH           the v29 autopilot .py (auto-found next to this script)
  --shots 100          per-task shots (same as existing Table IX row)
  --max-usd 300        abort before submitting if the estimate exceeds this
  --seed 20260702      reproducibility (sampling + item choice)

Outputs
-------
  qpu_multirow_percircuit.csv   one line per candidate instance (incl. task id,
                                sampled row indices, item mapping -> provenance)
  qpu_multirow_rows.csv         per-group aggregate = one Table IX row each
  qpu_multirow_tableIX.tex      ready-to-paste LaTeX rows

Needs: qiskit, qiskit-braket-provider, amazon-braket-sdk (only for --submit).
NOTE  Braket pricing defaults below are from earlier runs -- verify current
      per-task/per-shot rates in the AWS console before --submit.
"""
from __future__ import annotations
import argparse, csv, glob, hashlib, importlib.util, math, os, random, sys, time

ITEMS = ["A", "B", "C"]

# ----------------------------------------------------------------- groups ---
def default_groups(with_ankaa: bool):
    g = [
        dict(row="Tic-Tac-Toe", file="tic-tac-toe.data", Ms=[5, 6],
             n_target=11, sigmas=[2, 3], device="forte"),
        dict(row="Car", file="car.data", Ms=[7, 8],
             n_target=11, sigmas=[2, 3], device="forte"),
    ]
    if with_ankaa:
        g.append(dict(row="Chess", file="kr-vs-kp.data", Ms=[4, 5],
                      n_target=10, sigmas=[2], device="ankaa"))
    return g

DEVICE_HINTS = {   # name substring for provider lookup, fallback ARN
    "forte": ("Forte", "arn:aws:braket:us-east-1::device/qpu/ionq/Forte-1"),
    "ankaa": ("Ankaa", "arn:aws:braket:us-west-1::device/qpu/rigetti/Ankaa-3"),
    "garnet": ("Garnet", "arn:aws:braket:eu-north-1::device/qpu/iqm/Garnet"),
}
# $ per task + $ per shot -- VERIFY against current AWS pricing before --submit.
PRICE = {"forte": (0.30, 0.08), "ankaa": (0.30, 0.0009), "garnet": (0.30, 0.00145)}

# ------------------------------------------------------------ data loading ---
def load_uci_transactions(path):
    """UCI .data (comma-separated categorical row) -> list of frozensets of
    'a{i}={v}' items. Every attribute column (incl. class) becomes items."""
    txs = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals = [v.strip() for v in line.split(",")]
            txs.append(frozenset(f"a{i}={v}" for i, v in enumerate(vals) if v != "?"))
    if not txs:
        raise SystemExit(f"no transactions parsed from {path}")
    return txs

def file_sha1(path, n=8):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()[:n]

# ------------------------------------------------------- instance sampling ---
def support(txs, itemset):
    return sum(1 for t in txs if set(itemset).issubset(t))

def _sample_once(all_txs, group, rng, out, ds_id, attempt):
    M = group["Ms"][attempt % len(group["Ms"])]      # alternate so both M appear
    sigma = rng.choice(group["sigmas"])
    idx = rng.sample(range(len(all_txs)), M)
    sample = [all_txs[i] for i in idx]
    freq = {}
    for t in sample:
        for it in t:
            freq[it] = freq.get(it, 0) + 1
    # Item projection: anchor on the two sampled rows with the largest
    # intersection, so at least one candidate pair truly co-occurs (support
    # >= 2), then add a third item from outside the intersection so labels
    # straddle sigma. Item choice is a projection design decision; the sampled
    # rows and the item mapping are recorded in the CSV for provenance.
    ok = lambda it: freq[it] < M or M <= 4          # drop always-present items
    best, binter = None, []
    for a in range(len(sample)):
        for b in range(a + 1, len(sample)):
            inter = [it for it in (sample[a] & sample[b]) if ok(it)]
            if len(inter) > len(binter):
                best, binter = (a, b), inter
    if len(binter) < 2:
        return ds_id
    binter.sort(key=lambda it: (abs(freq[it] - (sigma + 1)), rng.random()))
    i1, i2 = binter[:2]
    solo = [it for it in (sample[best[0]] | sample[best[1]]) if ok(it) and it not in (i1, i2)]
    solo += [it for it, c in freq.items() if ok(it) and it not in solo and it not in (i1, i2)]
    if not solo:
        return ds_id
    solo.sort(key=lambda it: (abs(freq[it] - sigma), rng.random()))
    chosen = [i1, i2, solo[0]]
    return _emit(group, rng, out, ds_id, M, sigma, idx, sample, chosen)


def sample_group_instances(all_txs, group, seed):
    """Sample datasets of M rows until n_target candidate instances collected.
    Retries with shifted seeds until BOTH labels appear with a sane mix
    (minority class >= max(2, n_target//4))."""
    for retry in range(30):
        rng = random.Random(seed + retry * 7919)
        out, ds_id, attempts = [], 0, 0
        while len(out) < group["n_target"] and attempts < 4000:
            attempts += 1
            ds_id = _sample_once(all_txs, group, rng, out, ds_id, attempts)
        nf = sum(i["ideal"] for i in out)
        minority = min(nf, len(out) - nf)
        if len(out) >= group["n_target"] and minority >= max(2, group["n_target"] // 4):
            return out
    raise SystemExit(f"[{group['row']}] could not assemble a balanced group; "
                     f"relax Ms/sigmas or change --seed")


def _emit(group, rng, out, ds_id, M, sigma, idx, sample, chosen):
    mapping = dict(zip(ITEMS, sorted(chosen)))
    proj = [frozenset(a for a, real in mapping.items() if real in t) for t in sample]
    # QFM level-1 -> C2, exactly like the original toy flow
    F1 = [a for a in ITEMS if support(proj, [a]) >= sigma]
    C2 = [(F1[i], F1[j]) for i in range(len(F1)) for j in range(i + 1, len(F1))]
    if not C2:
        return ds_id
    ds_id += 1
    for cand in C2:
        if len(out) >= group["n_target"]:
            break
        sup = support(proj, cand)
        out.append(dict(row=group["row"], dataset_id=ds_id, M=M, sigma=sigma,
                        candidate="".join(cand), cand_tuple=cand, txs=proj,
                        support=sup, ideal=int(sup >= sigma),
                        sampled_rows=";".join(map(str, sorted(idx))),
                        item_map=";".join(f"{k}={v}" for k, v in mapping.items())))
    return ds_id

# ------------------------------------------------------------ v29 circuit ---
def find_v29(explicit):
    if explicit:
        return explicit
    here = os.path.dirname(os.path.abspath(__file__))
    for d in (here, os.getcwd(), os.path.expanduser("~/SageMaker")):
        c = [p for p in sorted(glob.glob(os.path.join(
             d, "qfm_paper_toy_qiskit_v29_braket_sweep6_qpu_autopilot*.py")))
             if "cancel" not in p and "(1)" not in p and "(2)" not in p]
        if c:
            return c[0]
    raise SystemExit("v29 autopilot .py not found; pass --v29 <path>")

def load_v29(path):
    spec = importlib.util.spec_from_file_location("v29mod", path)
    m = importlib.util.module_from_spec(spec)
    sys.modules["v29mod"] = m
    spec.loader.exec_module(m)
    return m

# --------------------------------------------------------------- backends ---
def get_backend(kind):
    from qiskit_braket_provider import BraketProvider
    name_hint, arn = DEVICE_HINTS[kind]
    prov = BraketProvider()
    exact = {"forte": ["Forte 1", "Forte Enterprise 1"],
             "ankaa": ["Ankaa-3"], "garnet": ["Garnet", "Emerald"]}[kind]
    for nm in exact:                       # cheap path: exact device names
        try:
            return prov.get_backend(nm)
        except Exception:
            pass
    try:
        cands = [b for b in prov.backends() if name_hint.lower() in b.name.lower()]
        if cands:
            return cands[0]
    except Exception as e:
        print(f"[warn] provider listing failed ({e}); trying ARN {arn}")
    try:
        from qiskit_braket_provider import BraketAwsBackend  # provider >=0.4 naming
        return BraketAwsBackend(arn=arn)
    except Exception as e:
        # NOTE: if this raises AttributeError on device properties (None), the
        # installed amazon-braket-schemas is too old for the device's current
        # capabilities document:  pip install -U amazon-braket-sdk amazon-braket-schemas
        from braket.aws import AwsDevice
        from qiskit_braket_provider import AWSBraketBackend  # older naming (positional device)
        return AWSBraketBackend(AwsDevice(arn))

# --------------------------------------------- RAW braket fallback (Plan B) ---
# Used when qiskit-braket-provider cannot parse the device capabilities
# ("Unable to determine device capabilities" -> properties=None). Bypasses the
# provider entirely: qiskit-transpile locally, convert with the provider's
# adapter, submit via braket-sdk AwsDevice.run. Restricted to all-to-all IonQ
# devices (no routing => qubit indices preserved). Before any QPU spend, the
# flag-bit mapping is validated on the SV1 simulator (costs cents).
RAW_OK = {"forte"}
RAW_BASIS = ["rz", "ry", "rx", "h", "cx"]

def raw_to_braket(tqc):
    try:
        from qiskit_braket_provider.providers.adapter import to_braket
        return to_braket(tqc)
    except ImportError:
        from qiskit_braket_provider.providers.adapter import (
            convert_qiskit_to_braket_circuit)
        return convert_qiskit_to_braket_circuit(tqc)

def raw_flag_pos(tqc, flag_idx):
    lay = getattr(tqc, "layout", None)
    if lay is None:
        return flag_idx
    try:
        return lay.final_index_layout()[flag_idx]
    except Exception:
        return flag_idx

def raw_p_flag1(counts, pos, orient):
    tot = sum(counts.values()) or 1
    if orient == "rev":
        return sum(v for k, v in counts.items() if k[-1 - pos] == "1") / tot
    return sum(v for k, v in counts.items() if k[pos] == "1") / tot

def raw_validate_on_sv1(samples, shots=1000):
    """samples = [(tqc, flag_pos, ideal_label), ...] with both labels present.
    Determines the bitstring orientation and proves the pipeline end-to-end on
    the SV1 simulator BEFORE any QPU money is spent."""
    from braket.aws import AwsDevice
    sv1 = AwsDevice("arn:aws:braket:::device/quantum-simulator/amazon/sv1")
    obs = []
    for tqc, pos, ideal in samples:
        counts = sv1.run(raw_to_braket(tqc), shots=shots).result().measurement_counts
        obs.append((counts, pos, ideal))
    for orient in ("idx", "rev"):
        if all(abs(raw_p_flag1(c, p, orient) - i) < 0.1 for c, p, i in obs):
            print(f"[raw] SV1 validation PASSED (bit orientation = {orient})")
            return orient
    for c, p, i in obs:
        print(f"[raw]   ideal={i}  p_idx={raw_p_flag1(c,p,'idx'):.3f}  p_rev={raw_p_flag1(c,p,'rev'):.3f}")
    raise SystemExit("[raw] SV1 validation FAILED under both bit orders -- aborting before any QPU spend")

# ----------------------------------------------------- deep-M Aer supplement ---
def find_benchmark_module():
    here = os.path.dirname(os.path.abspath(__file__))
    for d in (here, os.path.join(here, "QFM_error_exp"), os.getcwd()):
        p = os.path.join(d, "qfm_qsv_aer_benchmark.py")
        if os.path.exists(p):
            spec = importlib.util.spec_from_file_location("qsvbench", p)
            m = importlib.util.module_from_spec(spec)
            sys.modules["qsvbench"] = m
            spec.loader.exec_module(m)
            return m
    raise SystemExit("qfm_qsv_aer_benchmark.py not found (needed for --aer-supplement); "
                     "put this script next to it or inside QFM_error_exp/")


def gen_balanced_instances(counts, sigmas, p_item, seed):
    """Random ABC datasets like the 300-circuit benchmark, but with per-M
    label quotas so every M-bin is label-BALANCED (fixes single-class deep bins)."""
    rng = random.Random(seed)
    items = ["A", "B", "C"]
    pairs = [("A", "B"), ("A", "C"), ("B", "C")]
    out = []
    for M, ndat in counts.items():
        n_per_label = (3 * ndat) // 2
        quota = {1: n_per_label, 0: 3 * ndat - n_per_label}
        attempts = 0
        while (quota[0] > 0 or quota[1] > 0) and attempts < 60000:
            attempts += 1
            txs = []
            for _ in range(M):
                while True:
                    t = frozenset(it for it in items if rng.random() < p_item)
                    if t:
                        break
                txs.append(t)
            sigma = rng.choice(sigmas)
            for cand in pairs:
                sup = sum(1 for t in txs if set(cand).issubset(t))
                lab = int(sup >= sigma)
                if quota[lab] <= 0:
                    continue
                quota[lab] -= 1
                out.append(dict(M=M, sigma=int(sigma), candidate="".join(cand),
                                cand_tuple=cand, txs=txs, support=sup, ideal=lab))
        if quota[0] > 0 or quota[1] > 0:
            raise SystemExit(f"[supplement] could not fill balanced quotas at M={M} "
                             f"(left {quota}); adjust --supp-p-item/--supp-sigmas")
    return out


def run_aer_supplement(args, build_fn):
    bench = find_benchmark_module()
    counts = bench.parse_counts(args.supp_counts)
    sigmas = [int(x) for x in args.supp_sigmas.split(",") if x.strip()]
    insts = gen_balanced_instances(counts, sigmas, args.supp_p_item, args.seed)
    nf = sum(i["ideal"] for i in insts)
    print(f"[supplement] {len(insts)} balanced instances  counts={counts}  "
          f"freq/infq={nf}/{len(insts)-nf}  p_item={args.supp_p_item}")
    rows = []
    for preset in bench.PRESET_ORDER:
        e1, e2, ero = bench.NOISE_PRESETS[preset]
        nm = None if e2 == 0 and e1 == 0 and ero == 0 else bench.build_noise_model(e1, e2, ero)
        for k, ins in enumerate(insts):
            p, nq = bench.measure_p_flag1(build_fn, ins["txs"], ins["cand_tuple"], ins["sigma"],
                                          nm, args.supp_shots, args.seed + k)
            rows.append(dict(preset=preset, two_q_err=e2, M=ins["M"], sigma=ins["sigma"],
                             candidate=ins["candidate"], support=ins["support"],
                             label="frequent" if ins["ideal"] else "infrequent",
                             ideal=ins["ideal"], p_flag1=round(p, 4),
                             correct=int((p > 0.5) == bool(ins["ideal"])), qubits=nq))
        pm = {}
        for r in [r for r in rows if r["preset"] == preset]:
            c, t = pm.get(r["M"], (0, 0))
            pm[r["M"]] = (c + r["correct"], t + 1)
        print(f"  [{preset:13s}] per-M: " +
              "  ".join(f"M{M}:{c}/{t}" for M, (c, t) in sorted(pm.items())))
    with open(f"{args.out_prefix}_aer_supplement_percircuit.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    perm = {}
    for r in rows:
        key = (r["preset"], r["two_q_err"], r["M"])
        c, t = perm.get(key, (0, 0))
        perm[key] = (c + r["correct"], t + 1)
    with open(f"{args.out_prefix}_aer_supplement_perM.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["preset", "two_q_err", "M", "correct", "total", "correctness_pct"])
        for (pr, e2, M), (c, t) in sorted(perm.items()):
            w.writerow([pr, e2, M, c, t, round(100 * c / t, 1)])
    print(f"[done] wrote {args.out_prefix}_aer_supplement_percircuit.csv / _perM.csv "
          f"(balanced deep-M data; merge with qsv_bench_perM.csv when updating the curve)")


# ------------------------------------------------------------------- main ---
def main():
    ap = argparse.ArgumentParser(description="QFM Table-IX multi-row QPU validation")
    ap.add_argument("--v29", default=None)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--shots", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260702)
    ap.add_argument("--with-ankaa", action="store_true")
    ap.add_argument("--submit", action="store_true", help="actually run on QPUs (default: dry run)")
    ap.add_argument("--max-usd", type=float, default=300.0)
    ap.add_argument("--c0-order", choices=["right", "left"], default="right",
                    help="bit position of c[0] in counts keys (Forte-calibrated default: right)")
    ap.add_argument("--out-prefix", default="qpu_multirow")
    # ---- deep-M Aer supplement (local simulator, no AWS cost) ----
    ap.add_argument("--aer-supplement", action="store_true",
                    help="instead of QPU rows: generate BALANCED deep-M instances and run them "
                         "on local Qiskit Aer under the benchmark noise presets (fixes the "
                         "small-N single-class M>=10 bins of the 300-circuit benchmark)")
    ap.add_argument("--supp-counts", default="8:20,10:20,12:20",
                    help="M:datasets for the supplement (balanced labels enforced per M)")
    ap.add_argument("--supp-sigmas", default="2,3")
    ap.add_argument("--supp-p-item", type=float, default=0.50)
    ap.add_argument("--supp-shots", type=int, default=4000)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = args.data_dir or next((d for d in (
        os.path.join(here, "data_raw"),
        os.path.join(here, "fim_experiment_pack", "fim_experiment_pack", "data_raw"),
        os.path.join(os.path.expanduser("~/SageMaker"), "data_raw")) if os.path.isdir(d)), None)
    if not data_dir:
        raise SystemExit("data_raw folder not found; pass --data-dir")

    v29_path = find_v29(args.v29)
    print(f"[v29 circuit] {v29_path}")
    v29 = load_v29(v29_path)
    build_fn = v29.build_threshold_oracle_circuit

    if args.aer_supplement:
        return run_aer_supplement(args, build_fn)

    from qiskit import transpile
    groups = default_groups(args.with_ankaa)
    plan, est_cost = [], 0.0
    for gi, g in enumerate(groups):
        path = os.path.join(data_dir, g["file"])
        txs = load_uci_transactions(path)
        insts = sample_group_instances(txs, g, seed=args.seed + gi * 101)
        per_task, per_shot = PRICE[g["device"]]
        for ins in insts:
            qc, flag_idx = build_fn(ins["txs"], ITEMS, ins["cand_tuple"], int(ins["sigma"]))
            tl = transpile(qc, basis_gates=["rz", "sx", "x", "cx"], optimization_level=1,
                           seed_transpiler=args.seed)
            ins.update(qubits=qc.num_qubits, flag_idx=flag_idx,
                       logical_depth=tl.depth(), device=g["device"],
                       data_file=g["file"], data_sha1=file_sha1(path))
            est_cost += per_task + per_shot * args.shots
            plan.append(ins)
        n_f = sum(i["ideal"] for i in insts)
        print(f"[{g['row']:12s}] {len(insts)} instances  M={sorted(set(i['M'] for i in insts))}  "
              f"freq/infreq={n_f}/{len(insts)-n_f}  qubits={sorted(set(i['qubits'] for i in insts))}  "
              f"max logical depth={max(i['logical_depth'] for i in insts)}  device={g['device']}")
    print(f"[estimate] {len(plan)} tasks x {args.shots} shots  ~= ${est_cost:.2f} "
          f"(rates {PRICE}; VERIFY current AWS pricing)")
    if est_cost > args.max_usd:
        raise SystemExit(f"estimated ${est_cost:.2f} > --max-usd {args.max_usd}; aborting")

    # classical self-check: labels recomputed independently
    for ins in plan:
        assert ins["ideal"] == int(support(ins["txs"], ins["cand_tuple"]) >= ins["sigma"])
    print("[verify] all labels re-checked classically: OK")

    fields = ["row", "device", "data_file", "data_sha1", "dataset_id", "sampled_rows", "item_map",
              "M", "sigma", "candidate", "support", "label", "ideal", "qubits", "logical_depth",
              "backend_depth", "shots", "p_flag1", "correct", "task_id", "wall_s"]
    for ins in plan:
        ins.setdefault("label", "frequent" if ins["ideal"] else "infrequent")
        for k in ("backend_depth", "p_flag1", "correct", "task_id", "wall_s"):
            ins.setdefault(k, "")
        ins["shots"] = args.shots

    if not args.submit:
        write_outputs(plan, groups, args, dry=True)
        print("\nDRY RUN complete. Re-run with --submit to execute on the QPUs.")
        return

    # ---------------------------------------------------------- submission ---
    backends, raw_dev, raw_orient = {}, {}, None
    for g in groups:
        d = g["device"]
        if d in backends or d in raw_dev:
            continue
        print(f"[backend] resolving {d} ...")
        try:
            backends[d] = get_backend(d)
            print(f"[backend] {d} -> {backends[d].name}")
        except Exception as e:
            if d in RAW_OK:
                from braket.aws import AwsDevice
                raw_dev[d] = AwsDevice(DEVICE_HINTS[d][1])
                print(f"[backend] {d}: provider failed ({type(e).__name__}: {e}); "
                      f"switching to RAW braket submission (SV1-validated first)")
            else:
                raise

    def _raw_transpiled(ins):
        qc, _ = build_fn(ins["txs"], ITEMS, ins["cand_tuple"], int(ins["sigma"]))
        tqc = transpile(qc, basis_gates=RAW_BASIS, optimization_level=1,
                        seed_transpiler=args.seed)
        tqc.global_phase = 0  # Braket OpenQASM rejects GPhase; irrelevant for measurement stats
        return tqc, raw_flag_pos(tqc, ins["flag_idx"])

    if raw_dev and raw_orient is None:
        # one frequent + one infrequent instance from a raw-mode device group
        raw_ins = [i for i in plan if i["device"] in raw_dev]
        val = [next(i for i in raw_ins if i["ideal"] == 1),
               next(i for i in raw_ins if i["ideal"] == 0)]
        print("[raw] validating flag-bit mapping on SV1 (costs cents) ...")
        raw_orient = raw_validate_on_sv1(
            [(_raw_transpiled(i)[0], _raw_transpiled(i)[1], i["ideal"]) for i in val])

    for k, ins in enumerate(plan):
        t0 = time.time()
        if ins["device"] in raw_dev:
            tqc, pos = _raw_transpiled(ins)
            ins["backend_depth"] = tqc.depth()
            task = raw_dev[ins["device"]].run(raw_to_braket(tqc), shots=args.shots)
            ins["task_id"] = task.id
            print(f"  [{k+1}/{len(plan)}] {ins['row']} {ins['candidate']} M{ins['M']} s{ins['sigma']} "
                  f"-> {ins['task_id']} (depth {ins['backend_depth']}, raw)", flush=True)
            counts = task.result().measurement_counts
            p = raw_p_flag1(counts, pos, raw_orient)
        else:
            be = backends[ins["device"]]
            qc, flag_idx = build_fn(ins["txs"], ITEMS, ins["cand_tuple"], int(ins["sigma"]))
            meas, _ = v29.attach_measure_all_with_flag_to_c0(qc, ins["flag_idx"])
            tqc = transpile(meas, backend=be, optimization_level=1, seed_transpiler=args.seed)
            ins["backend_depth"] = tqc.depth()
            job = be.run(tqc, shots=args.shots)
            ins["task_id"] = getattr(job, "job_id", lambda: "")()
            print(f"  [{k+1}/{len(plan)}] {ins['row']} {ins['candidate']} M{ins['M']} s{ins['sigma']} "
                  f"-> {ins['task_id']} (depth {ins['backend_depth']})", flush=True)
            counts = job.result().get_counts()
            tot = sum(counts.values()) or 1
            pos = -1 if args.c0_order == "right" else 0
            p = sum(v for kk, v in counts.items() if kk.replace(" ", "")[pos] == "1") / tot
        ins["wall_s"] = round(time.time() - t0, 1)
        ins["p_flag1"] = round(p, 4)
        ins["correct"] = int((p > 0.5) == bool(ins["ideal"]))
        print(f"        p(flag=1)={p:.3f}  ideal={ins['ideal']}  "
              f"{'OK' if ins['correct'] else 'MISMATCH'}  ({ins['wall_s']}s)", flush=True)
        write_outputs(plan, groups, args, dry=False)  # incremental save
    write_outputs(plan, groups, args, dry=False)
    print("\n[done] see qpu_multirow_rows.csv / qpu_multirow_tableIX.tex")

def write_outputs(plan, groups, args, dry):
    fields = ["row", "device", "data_file", "data_sha1", "dataset_id", "sampled_rows", "item_map",
              "M", "sigma", "candidate", "support", "label", "ideal", "qubits", "logical_depth",
              "backend_depth", "shots", "p_flag1", "correct", "task_id", "wall_s"]
    with open(f"{args.out_prefix}_percircuit.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in plan:
            w.writerow(r)
    rows, tex = [], []
    for g in groups:
        sub = [r for r in plan if r["row"] == g["row"]]
        done = [r for r in sub if r["correct"] != ""]
        qs = sorted(set(r["qubits"] for r in sub))
        q_str = f"{qs[0]}" if len(qs) == 1 else f"{qs[0]}--{qs[-1]}"
        depths = [r["backend_depth"] or r["logical_depth"] for r in sub]
        corr = (f"{100.0*sum(r['correct'] for r in done)/len(done):.0f}\\%" if done else "--")
        rows.append(dict(row=g["row"], device=g["device"], n=len(sub), qubits=q_str,
                         max_depth=max(depths), correct=corr.replace("\\%", "%")))
        dev_name = {"forte": "IonQ Forte 1", "ankaa": "Rigetti Ankaa-3", "garnet": "IQM Garnet"}[g["device"]]
        tex.append(f"{dev_name} ({g['row']}) & {len(sub)} & {q_str} & {max(depths)} & {corr} \\\\")
    with open(f"{args.out_prefix}_rows.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    with open(f"{args.out_prefix}_tableIX.tex", "w", encoding="utf-8") as f:
        f.write("% paste into tab:braket_qpu_summary (existing Mushroom row stays first)\n")
        f.write("% Backend (sample dataset) & Tested instances & Logical qubits & Max depth & Correctness\n")
        for t in tex:
            f.write(t + "\n")
    if dry:
        print(f"[dry] wrote {args.out_prefix}_percircuit.csv (plan), _rows.csv, _tableIX.tex (placeholders)")

if __name__ == "__main__":
    main()
