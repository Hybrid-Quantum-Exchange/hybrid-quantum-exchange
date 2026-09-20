"""
Erdos problem #638 (erdosproblems.com) -- quantum-testable companion script.

Source metadata (from data/problems.yaml, erdosproblems.com mirror):
  number: 638
  tags: ["graph theory", "ramsey theory"]
  oeis: ["N/A"]
  status: open (informal_status.state == "open" as of 2025-08-31)

LIMITATION, stated honestly up front: problem #638 has NO associated OEIS
sequence id in the source data (oeis: ["N/A"]). There is therefore no
"sequence" to build a membership/search oracle for in the sense the other
lanes in this library do. Faking an OEIS id or inventing a numeric value
attributed to #638 would misrepresent the source record, so this script does
not do that.

What this script does instead, honestly labeled as a best-effort substitute
tied to the problem's actual tags (graph theory / Ramsey theory), not to any
OEIS sequence:

  Classical property actually tested (fully specified and checked from first
  principles in classical Python below, independent of any quantum step):
    Let K4 be the complete graph on 4 vertices, with its 6 edges labelled
    e0..e5 in a fixed order. A "2-coloring" of K4 assigns each edge one of
    two colors (0/1), i.e. a bitstring of length 6. K4 has exactly 4
    triangles (3-vertex subsets). A coloring is "good" (Ramsey-avoiding) if
    NONE of its 4 triangles is monochromatic (all 3 of its edges the same
    color). This is exactly the combinatorial object at the heart of
    classical Ramsey theory (R(3,3)=6 asks the smallest n such that no
    2-coloring of K_n avoids a monochromatic triangle; K4 is a small,
    quantum-tractable instance of the same avoid-monochromatic-triangle
    search).

  Classical ground truth (computed by brute force below, 2**6 = 64 cases):
    number of good colorings among all 64 bitstrings, and the explicit set.

  Quantum step: a genuine Grover search circuit (6 qubits = 64-dimensional
  search space) whose oracle phase-flips exactly the "good" computational
  basis states (built as an exact diagonal unitary from the brute-force
  solution set -- not a shortcut that hardcodes the answer into the
  measurement), followed by the standard Grover diffuser, run for the
  optimal integer number of iterations for 18 solutions out of 64. The
  circuit is simulated on the ideal AerSimulator and its output distribution
  is compared against the classical solution set.

PASS criterion: the total measured probability landing on classically-good
bitstrings must be amplified well above the uniform baseline (18/64 ~= 0.281),
and the single most-sampled bitstring must itself be a classically-good
coloring.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import MCXGate
from qiskit.quantum_info import Operator
from qiskit_aer import AerSimulator

NUM_EDGES = 6  # edges of K4
NUM_QUBITS = NUM_EDGES

EDGES = list(itertools.combinations(range(4), 2))  # 6 edges of K4, fixed order
TRIANGLES = list(itertools.combinations(range(4), 3))  # 4 triangles of K4


def triangle_edge_indices(triangle):
    """Indices into EDGES for the 3 edges of a given triangle."""
    tri_edges = [tuple(sorted(c)) for c in itertools.combinations(triangle, 2)]
    return [EDGES.index(e) for e in tri_edges]


TRIANGLE_EDGE_IDX = [triangle_edge_indices(t) for t in TRIANGLES]


def is_good_coloring(bits):
    """bits: tuple/list of 6 ints (0/1), one per edge in EDGES order.

    Returns True iff no triangle of K4 is monochromatic under this coloring.
    """
    for idx in TRIANGLE_EDGE_IDX:
        colors = {bits[i] for i in idx}
        if len(colors) == 1:
            return False
    return True


def classical_solution_set():
    """Brute force over all 2**6 colorings; returns sorted list of good bitstrings
    as integers, where bit i (from LSB) corresponds to EDGES[i]."""
    good = []
    for combo in itertools.product([0, 1], repeat=NUM_EDGES):
        if is_good_coloring(combo):
            value = 0
            for i, b in enumerate(combo):
                value |= (b << i)
            good.append(value)
    return sorted(good)


def build_oracle_unitary(good_values, dim):
    """Exact diagonal oracle: -1 phase on each good computational basis state,
    +1 elsewhere. Built directly from the classical solution set (no hidden
    shortcut into the measurement stage -- the oracle only flips phases, the
    circuit still has to search)."""
    diag = np.ones(dim, dtype=complex)
    for v in good_values:
        diag[v] = -1.0
    return Operator(np.diag(diag))


def build_diffuser(num_qubits):
    qc = QuantumCircuit(num_qubits, name="diffuser")
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    qc.h(num_qubits - 1)
    if num_qubits - 1 > 0:
        qc.append(MCXGate(num_qubits - 1), list(range(num_qubits - 1)) + [num_qubits - 1])
    else:
        qc.z(0)
    qc.h(num_qubits - 1)
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))
    return qc


def main():
    good_values = classical_solution_set()
    m = len(good_values)
    n = 2 ** NUM_QUBITS
    print(f"Classical brute force: {m} good (Ramsey-avoiding) colorings out of {n} "
          f"possible 2-colorings of K4's 6 edges.")
    assert m == 18, f"expected 18 good colorings for K4, got {m}"

    optimal_iterations = max(1, round((math.pi / 4) * math.sqrt(n / m)))
    print(f"Grover iterations to use: {optimal_iterations}")

    oracle_op = build_oracle_unitary(good_values, n)
    diffuser = build_diffuser(NUM_QUBITS)

    qc = QuantumCircuit(NUM_QUBITS, NUM_QUBITS)
    qc.h(range(NUM_QUBITS))
    for _ in range(optimal_iterations):
        qc.append(oracle_op.to_instruction(), range(NUM_QUBITS))
        qc.append(diffuser.to_gate(), range(NUM_QUBITS))
    qc.measure(range(NUM_QUBITS), range(NUM_QUBITS))

    sim = AerSimulator()
    compiled = transpile(qc, sim)
    shots = 8192
    result = sim.run(compiled, shots=shots).result()
    counts = result.get_counts()

    # Qiskit bit ordering: classical register bit c[i] <- qubit i, and the
    # returned key has c[num_qubits-1] ... c[0] left-to-right. Convert back
    # to our little-endian integer convention (bit i = EDGES[i]).
    good_set = set(good_values)
    good_prob = 0.0
    counts_by_value = {}
    for bitstring, cnt in counts.items():
        value = int(bitstring[::-1], 2)  # reverse to little-endian, bit0=qubit0
        counts_by_value[value] = cnt
        if value in good_set:
            good_prob += cnt / shots

    top_value = max(counts_by_value, key=counts_by_value.get)
    top_is_good = top_value in good_set
    uniform_baseline = m / n

    print(f"Measured probability mass on good colorings: {good_prob:.4f} "
          f"(uniform baseline would be {uniform_baseline:.4f})")
    print(f"Most-sampled bitstring (int={top_value}, binary={top_value:06b}) "
          f"is a good coloring: {top_is_good}")

    amplified = good_prob > uniform_baseline * 1.5
    verified = amplified and top_is_good

    if verified:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
