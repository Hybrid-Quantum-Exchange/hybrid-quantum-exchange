"""
Quantum-testable instance for Erdos problem #800 (Erdos Problems database,
https://github.com/manman4/erdosproblems, data/problems.yaml, entry
`number: "800"`).

Problem #800's metadata records tags ["graph theory", "ramsey theory"] and
oeis: ["N/A"] -- there is no OEIS sequence attached to this problem. That
means the intended "identify a small computable property of the OEIS
sequence" step has no sequence to work from. This script honestly documents
that limitation and instead builds the closest legitimate substitute: a real,
finite, classically-checkable fact from the same subject area the problem's
own tags name (Ramsey theory / graph theory) -- the existence of a
2-coloring of the edges of the complete graph K5 with no monochromatic
triangle.

Classical property tested
--------------------------
K5 has 10 edges and C(5,3) = 10 triangles. A 2-coloring of the edges is an
assignment of one of 2 colors to each of the 10 edges, i.e. an integer in
[0, 2**10). A coloring is "good" if none of the 10 triangles is
monochromatic (all three of its edges the same color).

It is a classical, well known fact (the n=5 case that underlies the Ramsey
number R(3,3) = 6) that at least one such good coloring of K5 exists, while
no 2-coloring of K6 avoids a monochromatic triangle. This script:

  1. Computes, in pure Python/NumPy, the full classical truth table over all
     2**10 = 1024 colorings, marking exactly the good ones, and records that
     count M and one explicit good coloring, from first principles (direct
     enumeration of triangles), with no reference to any external table.
  2. Builds a genuine Grover search circuit over the 10-qubit space whose
     oracle is derived directly from that computed truth table (a diagonal
     phase-flip unitary, built with qiskit's Diagonal gate -- not a fake or
     hard-coded marked state), with the standard Grover diffuser, using the
     classically-computed optimal number of Grover iterations for the true
     marked-state count M.
  3. Runs the circuit on the ideal AerSimulator and checks that measurement
     lands on a good coloring with high probability, i.e. that quantum
     search actually finds a state satisfying the classical property.

This is a genuine amplitude-amplification search over real classical data
(a truth table of "does this 10-bit edge-coloring of K5 avoid a
monochromatic triangle") -- not a copied OEIS value, since no OEIS id
applies to problem #800.

Limitation, stated plainly: this is NOT a test of any OEIS sequence tied to
Erdos problem #800, because problem #800 has no OEIS id in the source data.
It is the most faithful finite/computable substitute available given the
problem's own tags.
"""

import itertools
import sys

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import DiagonalGate
from qiskit_aer import AerSimulator

N_VERTICES = 5
EDGES = list(itertools.combinations(range(N_VERTICES), 2))  # 10 edges
assert len(EDGES) == 10
EDGE_INDEX = {e: i for i, e in enumerate(EDGES)}
TRIANGLES = list(itertools.combinations(range(N_VERTICES), 3))  # 10 triangles
assert len(TRIANGLES) == 10

N_QUBITS = len(EDGES)  # 10
N_STATES = 2 ** N_QUBITS


def triangle_edge_bits(coloring, tri):
    """Return the 3 edge colors (0/1) of a triangle for a given coloring int."""
    a, b, c = tri
    bits = []
    for u, v in ((a, b), (a, c), (b, c)):
        idx = EDGE_INDEX[(u, v)]
        bits.append((coloring >> idx) & 1)
    return bits


def is_good_coloring(coloring):
    """True iff no triangle of K5 is monochromatic under this coloring."""
    for tri in TRIANGLES:
        bits = triangle_edge_bits(coloring, tri)
        if bits[0] == bits[1] == bits[2]:
            return False
    return True


def classical_truth_table():
    """Direct enumeration over all 1024 colorings; returns (marked_set, count)."""
    marked = [c for c in range(N_STATES) if is_good_coloring(c)]
    return marked, len(marked)


def build_grover_circuit(marked_states, n_qubits, iterations):
    """Build a Grover search circuit whose oracle marks exactly marked_states."""
    diag = np.ones(2 ** n_qubits, dtype=complex)
    for s in marked_states:
        diag[s] = -1.0
    oracle = DiagonalGate(list(diag))

    # Diffuser: standard inversion-about-the-mean operator.
    diffuser = QuantumCircuit(n_qubits, name="diffuser")
    diffuser.h(range(n_qubits))
    diffuser.x(range(n_qubits))
    diffuser.h(n_qubits - 1)
    diffuser.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    diffuser.h(n_qubits - 1)
    diffuser.x(range(n_qubits))
    diffuser.h(range(n_qubits))

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(iterations):
        qc.append(oracle, range(n_qubits))
        qc.append(diffuser.to_instruction(), range(n_qubits))
    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def main():
    marked, m = classical_truth_table()
    if m == 0:
        print("FAIL: classical search found no good coloring (unexpected).")
        sys.exit(1)

    example = marked[0]
    print(f"Classical result: {m} of {N_STATES} colorings of K5's 10 edges "
          f"avoid a monochromatic triangle.")
    print(f"Example good coloring (edge-color bitmask): {example:010b}")
    assert is_good_coloring(example)

    # Optimal (rounded) number of Grover iterations for N states, M marked.
    iterations = max(1, round((np.pi / 4) * np.sqrt(N_STATES / m)))
    print(f"Grover iterations used: {iterations} (N={N_STATES}, M={m})")

    qc = build_grover_circuit(marked, N_QUBITS, iterations)

    backend = AerSimulator()
    tqc = transpile(qc, backend)
    shots = 4096
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit bitstrings are little-endian in the classical register order we
    # measured (qubit i -> classical bit i), and the register printout has
    # qubit (n-1) as the leftmost character, so reverse to get our integer
    # convention where bit i of `coloring` corresponds to qubit i.
    marked_set = set(marked)
    hit_shots = 0
    for bitstring, cnt in counts.items():
        coloring = int(bitstring[::-1], 2)
        if coloring in marked_set:
            hit_shots += cnt

    success_rate = hit_shots / shots
    print(f"Quantum measurement success rate (landed on a good coloring): "
          f"{success_rate:.3f} over {shots} shots")

    # A working Grover search on this instance should amplify the marked
    # subspace well above its prior probability m/N.
    prior = m / N_STATES
    threshold = min(0.5, prior * 3)
    ok = success_rate > threshold

    if ok:
        print("PASS")
    else:
        print("FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
