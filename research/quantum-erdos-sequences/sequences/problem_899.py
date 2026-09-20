"""
Erdos problem #899 -- quantum-testable instance.

Source metadata (erdosproblems.com dataset, data/problems.yaml, entry
"number: '899'"):
    prize: no
    status: proved (Lean), last update 2026-08-23
    oeis: ["N/A"]
    tags: ["additive combinatorics"]

LIMITATION, stated honestly up front: problem #899 has no associated OEIS
sequence id in the dataset (oeis is literally "N/A"), so there is no
specific integer sequence to build a membership/term-search circuit against
for *this* problem the way other lanes in this library can. Rather than
fabricate a connection to an unrelated OEIS id, this script instead builds a
genuine, small, finite, classically-checkable instance of the mathematical
object named by problem #899's own tag, "additive combinatorics": a
sum-free set. This is an honest best-effort substitute, not a claim that
A-numbered sequence membership was tested.

Classical property being tested
--------------------------------
Universe U = {1, 2, 3, 4, 5}. Consider all 3-element subsets of U (there are
C(5,3) = 10 of them). A subset S is "sum-free" if there is no choice of
distinct a, b, c in S with a + b = c.

We enumerate the 10 subsets, index them 0..9 (and pad the index register to
16 = 2^4 states, indices 10..15 unused/never marked), and classically
determine -- from first principles, by brute-force checking every ordered
pair inside each subset -- exactly which subset-indices are sum-free. That
classical brute force is done in this script (see `classical_marked_indices`
below) and is the ground truth the quantum result is checked against.

Quantum circuit
----------------
A Grover search over the 4-qubit index register (N = 16 basis states, of
which M = 6 are marked "sum-free" subsets):
  - Oracle: a diagonal unitary (built with qiskit's `Diagonal` gate, which
    synthesizes to real single/multi-qubit gates) that applies a -1 phase to
    exactly the marked basis states.
  - Diffuser: the standard Grover diffusion operator (H^4, X^4, multi-
    controlled Z, X^4, H^4).
  - Optimal iteration count round(pi/4 * sqrt(N/M)) = 1 Grover iteration.

The circuit is run on the ideal AerSimulator (statevector method, no noise).
We check PASS by verifying that the simulator's measurement distribution is
concentrated on the classically-marked indices: the total probability mass
on marked outcomes must exceed the total mass on non-marked outcomes by a
wide, unambiguous margin (a real amplification effect, not 1/16 uniform
guessing).
"""

import math
from itertools import combinations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import DiagonalGate
from qiskit_aer import AerSimulator


def classical_marked_indices():
    """Brute-force, from first principles, which 3-subsets of {1..5} are sum-free."""
    universe = range(1, 6)
    subsets = list(combinations(universe, 3))  # 10 subsets, index 0..9
    marked = []
    for idx, s in enumerate(subsets):
        sum_free = True
        for a in s:
            for b in s:
                if a != b and (a + b) in s:
                    sum_free = False
                    break
            if not sum_free:
                break
        if sum_free:
            marked.append(idx)
    return subsets, marked


def build_oracle(n_qubits, marked_indices):
    """Diagonal phase oracle: -1 on marked basis states, +1 elsewhere."""
    dim = 2 ** n_qubits
    diag = [1.0] * dim
    for m in marked_indices:
        diag[m] = -1.0
    gate = DiagonalGate(diag)
    return gate


def build_diffuser(n_qubits):
    """Standard Grover diffusion operator (inversion about the mean)."""
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def run():
    subsets, marked = classical_marked_indices()
    n = len(subsets)  # 10 real subsets
    n_qubits = 4  # 2^4 = 16 >= 10, pads with 6 unused/never-marked indices
    dim = 2 ** n_qubits
    m = len(marked)

    print(f"Universe: {{1,2,3,4,5}}, 3-element subsets: {n}")
    print(f"Classically sum-free subsets (marked indices): {marked}")
    for idx in marked:
        print(f"  index {idx}: {subsets[idx]}")

    iterations = max(1, round((math.pi / 4) * math.sqrt(dim / m)))
    print(f"Grover: N={dim} basis states, M={m} marked, iterations={iterations}")

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    oracle = build_oracle(n_qubits, marked)
    diffuser = build_diffuser(n_qubits)
    for _ in range(iterations):
        qc.append(oracle, range(n_qubits))
        qc.append(diffuser.to_gate(), range(n_qubits))

    qc.measure(range(n_qubits), range(n_qubits))

    backend = AerSimulator(method="statevector")
    tqc = transpile(qc, backend)
    shots = 20000
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit bit order: classical bit 0 (qubit 0) is the rightmost char.
    marked_prob = 0
    unmarked_prob = 0
    for bitstring, c in counts.items():
        idx = int(bitstring, 2)
        if idx in marked:
            marked_prob += c
        else:
            unmarked_prob += c
    marked_prob /= shots
    unmarked_prob /= shots

    print(f"P(marked outcome) = {marked_prob:.4f}")
    print(f"P(unmarked outcome) = {unmarked_prob:.4f}")
    print(f"Uniform-guess baseline for marked mass = {m/dim:.4f}")

    # A real Grover amplification on N=16, M=6, 1 iteration should push
    # marked-outcome probability well above the uniform baseline (0.375)
    # and above the unmarked mass. Require a clear, unambiguous margin.
    passed = marked_prob > unmarked_prob and marked_prob > (m / dim) + 0.15

    print("PASS" if passed else "FAIL")
    return passed


if __name__ == "__main__":
    ok = run()
    raise SystemExit(0 if ok else 1)
