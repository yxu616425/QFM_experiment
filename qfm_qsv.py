"""QSV threshold circuit with the offset included in the initial reduction.

Input qubits x_r indicate whether each merged transaction matches a candidate.
Given public multiplicities w_r, the circuit computes the flag
    q_th = [sum(w_r * x_r) >= sigma].
The reduction uses reversible 3-to-2 compressors followed by a carry-prefix
network. All workspace can be cleared by reversing the circuit.
"""
from qiskit import QuantumCircuit

def plan(weights, h, M):
    assert all((type(w) == int and w > 0 for w in weights))
    assert 1 <= h <= M and sum(weights) <= M
    b = M.bit_length()
    offset = (1 << b) - h
    heap = [0] * (b + 1)
    for w in weights:
        for j in range(b):
            heap[j] += w >> j & 1
    for j in range(b):
        heap[j] += offset >> j & 1
    initial = heap[:]
    layers = []
    while max(heap) > 2:
        groups = [n // 3 for n in heap]
        assert groups[b] == 0
        layers.append(groups)
        heap = [heap[j] % 3 + groups[j] + (groups[j - 1] if j else 0) for j in range(b + 1)]
    assert heap[b] <= 1
    counts = [w.bit_count() for w in weights]
    have = [1] * len(weights)
    fan = []
    while any((a < n for a, n in zip(have, counts))):
        adds = [min(a, n - a) for a, n in zip(have, counts)]
        fan.append(sum(adds))
        have = [a + d for a, d in zip(have, adds)]
    return dict(M=M, h=h, R=len(weights), width=b, total=sum(weights), offset=offset, weights=weights, initial_heap=initial, final_heap=heap, compressor_layers=layers, compressors=sum(map(sum, layers)), fanout_layers=fan, weight_set_bits=sum(counts), offset_set_bits=offset.bit_count(), row_copy_qubits=0, architecture='fused_offset_direct_input_balanced_carry_v1')

def prefix(heap):
    """Direct interface consists only of actual residual bits (absent = 0)."""
    b = len(heap) - 1
    cursor = 0
    inputs = []
    for n in heap:
        inputs.append(list(range(cursor, cursor + n)))
        cursor += n
    data = cursor
    ops = []

    def new():
        nonlocal cursor
        q = cursor
        cursor += 1
        return q
    flag = new()
    gp = []

    def add(name, *qs):
        if all((q is not None for q in qs)):
            ops.append((name, qs))
    for col in inputs[:b]:
        g, p = (new(), new())
        a = col[0] if col else None
        z = col[1] if len(col) > 1 else None
        add('ccx', a, z, g)
        add('cx', a, p)
        add('cx', z, p)
        gp.append((g, p))
    ops.append(('barrier', ()))
    while len(gp) > 1:
        nxt = []
        for j in range(0, len(gp), 2):
            if j + 1 == len(gp):
                nxt.append(gp[j])
                continue
            gl, pl = gp[j]
            gh, ph = gp[j + 1]
            g, p = (new(), new())
            add('cx', gh, g)
            add('ccx', ph, gl, g)
            add('ccx', ph, pl, p)
            nxt.append((g, p))
        gp = nxt
        ops.append(('barrier', ()))
    if inputs[b]:
        add('cx', inputs[b][0], flag)
    add('cx', gp[0][0], flag)
    qc = QuantumCircuit(cursor)
    for name, qs in ops:
        getattr(qc, name)(*qs)
    assert cursor - data == 4 * b - 1
    return (qc, flag, inputs)

def concrete(weights, h, M):
    p = plan(weights, h, M)
    R = len(weights)
    b = p['width']
    cursor = R
    heap = [[] for _ in range(b + 1)]
    groups = []
    for r, w in enumerate(weights):
        qs = [r] + list(range(cursor, cursor + w.bit_count() - 1))
        cursor += w.bit_count() - 1
        groups.append(qs)
        for j, q in zip([j for j in range(b) if w >> j & 1], qs):
            heap[j].append(q)
    ops = []
    have = 1
    while any((have < len(g) for g in groups)):
        for g in groups:
            for j in range(min(have, len(g) - have)):
                ops.append(('cx', (g[j], g[have + j])))
        have *= 2
    for j in range(b):
        if p['offset'] >> j & 1:
            heap[j].append(cursor)
            ops.append(('x', (cursor,)))
            cursor += 1
    for layer in p['compressor_layers']:
        nxt = [[] for _ in heap]
        for j, n in enumerate(layer):
            for z in range(n):
                a, d, e = heap[j][3 * z:3 * z + 3]
                s, c = (cursor, cursor + 1)
                cursor += 2
                ops += [('cx', (a, s)), ('cx', (d, s)), ('cx', (e, s)), ('ccx', (a, d, c)), ('ccx', (a, e, c)), ('ccx', (d, e, c))]
                nxt[j].append(s)
                nxt[j + 1].append(c)
            nxt[j].extend(heap[j][3 * n:])
        heap = nxt
    core, flag, _ = prefix(p['final_heap'])
    mapping = [q for col in heap for q in col]
    mapping += list(range(cursor, cursor + core.num_qubits - len(mapping)))
    cursor = max(mapping) + 1
    for ins in core.data:
        if ins.operation.name != 'barrier':
            ops.append((ins.operation.name, tuple((mapping[core.find_bit(q).index] for q in ins.qubits))))
    return (cursor, mapping[flag], ops, p)

def build_threshold_circuit(weights, sigma, M):
    """Return (forward circuit, flag index), with matching bits in qubits 0..R-1.

    M is the original transaction count, so sum(weights) must not exceed M.
    All other input qubits must start at zero. The input matching bits may be
    in superposition. The forward circuit leaves workspace for its inverse;
    build_phase_oracle also performs that inverse.
    """
    weights = list(weights)
    if type(M) is not int or M < 1:
        raise ValueError('M must be a positive integer')
    if type(sigma) is not int:
        raise ValueError('sigma must be an integer support count')
    if any(type(w) is not int or w < 1 for w in weights):
        raise ValueError('weights must be positive integers')
    if sum(weights) > M:
        raise ValueError('sum(weights) must not exceed M')
    if sigma <= 0 or sigma > sum(weights):
        qc = QuantumCircuit(len(weights) + 1, name='QSV threshold')
        flag = len(weights)
        if sigma <= 0:
            qc.x(flag)
        return qc, flag
    size, flag, operations, _ = concrete(weights, sigma, M)
    qc = QuantumCircuit(size, name='QSV threshold')
    for gate, wires in operations:
        getattr(qc, gate)(*wires)
    return qc, flag


def build_phase_oracle(weights, sigma, M):
    """Mark matching input states by a minus sign and return workspace to zero."""
    forward, flag = build_threshold_circuit(weights, sigma, M)
    qc = QuantumCircuit(forward.num_qubits, name='QSV phase oracle')
    qc.compose(forward, inplace=True)
    qc.z(flag)
    qc.compose(forward.inverse(), inplace=True)
    return qc


def evaluate_basis(circuit, input_bits):
    """Evaluate an X/CX/CCX circuit on a computational-basis input.

    This Boolean check avoids allocating a statevector for large circuits.
    It does not simulate amplitudes, noise, or execution time.
    """
    if len(input_bits) > circuit.num_qubits or any(b not in (0, 1) for b in input_bits):
        raise ValueError('input_bits must fit the circuit and contain only 0 or 1')
    state = list(input_bits) + [0] * (circuit.num_qubits - len(input_bits))
    for instruction in circuit.data:
        name = instruction.operation.name
        if name == 'barrier':
            continue
        if name not in ('x', 'cx', 'ccx'):
            raise ValueError('basis evaluation supports only X, CX, and CCX')
        wires = [circuit.find_bit(q).index for q in instruction.qubits]
        state[wires[-1]] ^= int(all(state[q] for q in wires[:-1]))
    return state


def main():
    # Figure 3: candidate {B, C} over six merged transactions.
    weights = [3, 3, 3, 2, 2, 2]
    matching_bits = [1, 1, 0, 1, 0, 1]
    sigma, M = 10, 15
    circuit, flag = build_threshold_circuit(weights, sigma, M)
    output = evaluate_basis(circuit, matching_bits)
    restored = evaluate_basis(circuit.inverse(), output)
    assert output[flag] == 1
    assert restored == matching_bits + [0] * (circuit.num_qubits - len(matching_bits))
    print('Figure 3: M=15, sigma=10, K_off=6')
    print('Basis-state check: q_th =', output[flag])
    print('Inverse check: input preserved and workspace cleared')


if __name__ == '__main__':
    main()
