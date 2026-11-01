import itertools
import random
import unittest

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from qfm_qsv import build_phase_oracle, build_threshold_circuit, evaluate_basis


class ThresholdTests(unittest.TestCase):
    def check_input(self, circuit, flag, weights, sigma, bits):
        actual = evaluate_basis(circuit, bits)
        expected = int(sum(w * x for w, x in zip(weights, bits)) >= sigma)
        self.assertEqual(actual[flag], expected)
        self.assertEqual(actual[:len(bits)], list(bits))
        restored = evaluate_basis(circuit.inverse(), actual)
        self.assertEqual(restored, list(bits) + [0] * (circuit.num_qubits - len(bits)))

    def test_exhaustive_small_inputs_and_boundaries(self):
        for size in range(1, 4):
            for weights in itertools.product(range(1, 4), repeat=size):
                for M in (sum(weights), sum(weights) + 3):
                    for sigma in range(0, M + 2):
                        circuit, flag = build_threshold_circuit(weights, sigma, M)
                        for bits in itertools.product((0, 1), repeat=size):
                            self.check_input(circuit, flag, weights, sigma, bits)

    def test_larger_multiplicities(self):
        rng = random.Random(20261102)
        for _ in range(100):
            weights = [rng.randint(1, 100) for _ in range(rng.randint(1, 30))]
            M = sum(weights) + rng.randint(0, 100)
            sigma = rng.randint(1, M)
            bits = [rng.randrange(2) for _ in weights]
            circuit, flag = build_threshold_circuit(weights, sigma, M)
            self.check_input(circuit, flag, weights, sigma, bits)

    def test_figure_3(self):
        weights = [3, 3, 3, 2, 2, 2]
        bits = [1, 1, 0, 1, 0, 1]
        circuit, flag = build_threshold_circuit(weights, 10, 15)
        self.check_input(circuit, flag, weights, 10, bits)
        self.assertEqual(evaluate_basis(circuit, bits)[flag], 1)

    def test_phase_and_clean_workspace_in_superposition(self):
        for weights, sigma, M in [([1, 2], 2, 3), ([1, 1, 1], 2, 3)]:
            oracle = build_phase_oracle(weights, sigma, M)
            qc = QuantumCircuit(oracle.num_qubits)
            qc.h(range(len(weights)))
            qc.compose(oracle, inplace=True)
            actual = Statevector.from_instruction(qc).data
            expected = np.zeros_like(actual)
            amplitude = 1 / np.sqrt(2 ** len(weights))
            for i in range(2 ** len(weights)):
                support = sum(w * ((i >> r) & 1) for r, w in enumerate(weights))
                expected[i] = (-1 if support >= sigma else 1) * amplitude
            np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_invalid_inputs(self):
        for args in [([0], 1, 1), ([2], 1, 1), ([1], 1.5, 2), ([1], 1, 0)]:
            with self.assertRaises(ValueError):
                build_threshold_circuit(*args)

    def test_empty_projection(self):
        for sigma in (-1, 0, 1, 2):
            circuit, flag = build_threshold_circuit([], sigma, 1)
            self.assertEqual(evaluate_basis(circuit, [])[flag], int(sigma <= 0))


if __name__ == '__main__':
    unittest.main()
