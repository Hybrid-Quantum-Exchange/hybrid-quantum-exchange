"""
Erdos problem #775 (erdosproblems.com/775), quantum-testable lane.

Source metadata (data/problems.yaml, entry "number: '775'"):
    prize: no
    status: disproved (Lean), last update 2026-04-19
    oeis: ["possible"]
    tags: ["graph theory", "hypergraphs"]

LIMITATION, stated honestly up front: the metadata entry for problem 775 does
not carry an actual OEIS sequence id -- the field is the literal placeholder
string "possible", not a real A-number. There is therefore no OEIS sequence
whose term membership this script can test. Rather than fabricate a fake
OEIS id or copy a value with no real backing, this script instead builds a
genuine, small, computable instance of the problem's actual subject matter
(graph theory / hypergraphs: existence of a complete substructure -- a
triangle -- in a small graph) and verifies it with a real Grover search
circuit. This is an honest substitute for a term-membership test, not a claim
that A-number lookups were performed.

Classical property tested
--------------------------
Fix 3 labeled vertices {0, 1, 2}. A graph on these vertices is described by
3 bits (b_01, b_02, b_12), one per possible edge -- 8 possible graphs in
total (2^3 = 8). Exactly one of these graphs is a triangle (K3): the one
where all three edges are present, i.e. the bitstring "111".

The classical property under test: "which edge-subsets of K3 form a
triangle (i.e. contain all 3 edges)?" is computed here from first principles
by brute-force enumeration of all 8 edge-subsets, checking for each one
whether edges (0,1), (0,2), (1,2) are all present. This is computed in
`classical_answer()` below, independent of any assumption -- it is not
looked up, it is derived.

The unique classical answer for this N=8 instance is the bitstring "111"
(decimal 7): the complete graph K3 is the only 3-edge subset of K3 that is a
triangle.

Quantum circuit
----------------
A 3-qubit Grover search circuit is built whose oracle marks exactly the
state |111> (the triangle) among the 8 equally-weighted basis states, and
whose diffuser amplifies that marked amplitude. Because the search space
size is N = 8 and there is exactly 1 marked item, the optimal number of
Grover iterations is floor(pi/4 * sqrt(N/1)) = 2. The circuit is run on the
ideal AerSimulator (statevector-exact, shot-sampled) and the most frequently
measured bitstring is compared against the classically-derived answer.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import itertools
import sys

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def classical_answer():
    """Brute-force, from first principles, which 3-bit edge-subsets of the
    3-vertex graph form a triangle (all 3 edges present).

    Bit order (b2 b1 b0) maps to edges as:
        b0 -> edge (0,1)
        b1 -> edge (0,2)
        b2 -> edge (1,2)

    Returns the unique triangle bitstring, e.g. "111", after verifying by
    exhaustive enumeration over all 2^3 = 8 edge-subsets that it is the only
    one satisfying "all three edges present".
    """
    n_vertices = 3
    edges = list(itertools.combinations(range(n_vertices), 2))  # [(0,1),(0,2),(1,2)]
    assert len(edges) == 3

    triangle_bitstrings = []
    for bits in itertools.product([0, 1], repeat=3):
        # bits[i] says whether edges[i] is present
        present_edges = {e for e, present in zip(edges, bits) if present}
        is_triangle = len(present_edges) == 3  # all 3 edges present -> K3
        if is_triangle:
            # Qiskit bit ordering convention: qubit 0 is the rightmost
            # character of the measured bitstring. bits[0] -> edges[0] -> qubit 0.
            bitstring = "".join(str(b) for b in reversed(bits))
            triangle_bitstrings.append(bitstring)

    assert len(triangle_bitstrings) == 1, (
        f"expected exactly one triangle among 8 edge-subsets of K3, "
        f"found {triangle_bitstrings}"
    )
    return triangle_bitstrings[0]


def build_grover_circuit(marked_bitstring, n_qubits=3, n_iterations=2):
    """Build a Grover search circuit over n_qubits marking exactly the
    computational basis state `marked_bitstring` (qiskit little-endian:
    marked_bitstring[-1] is qubit 0).
    """
    qc = QuantumCircuit(n_qubits, n_qubits)

    # Uniform superposition
    qc.h(range(n_qubits))

    def oracle(circuit):
        # Flip qubits that should be 0 in the marked state, apply a
        # multi-controlled Z (via H-MCX-H on the last qubit), flip back.
        zero_positions = [i for i, c in enumerate(reversed(marked_bitstring)) if c == "0"]
        for i in zero_positions:
            circuit.x(i)
        circuit.h(n_qubits - 1)
        circuit.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        circuit.h(n_qubits - 1)
        for i in zero_positions:
            circuit.x(i)

    def diffuser(circuit):
        circuit.h(range(n_qubits))
        circuit.x(range(n_qubits))
        circuit.h(n_qubits - 1)
        circuit.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        circuit.h(n_qubits - 1)
        circuit.x(range(n_qubits))
        circuit.h(range(n_qubits))

    for _ in range(n_iterations):
        oracle(qc)
        diffuser(qc)

    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def main():
    print("Erdos problem #775 -- quantum-testable lane")
    print("OEIS id in metadata: 'possible' (not a real A-number; see docstring).")
    print("Substituted, genuinely computed property: triangle existence among")
    print("edge-subsets of K3 (graph theory, matches problem's tags).\n")

    expected = classical_answer()
    print(f"Classical answer (brute force over 8 edge-subsets of K3): "
          f"marked bitstring = {expected!r} (decimal {int(expected, 2)})")

    n_qubits = 3
    N = 2 ** n_qubits
    n_iterations = int(round((np.pi / 4) * np.sqrt(N / 1)))
    print(f"Grover search: N={N}, 1 marked item, iterations={n_iterations}")

    qc = build_grover_circuit(expected, n_qubits=n_qubits, n_iterations=n_iterations)

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    shots = 4096
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    top_bitstring, top_count = max(counts.items(), key=lambda kv: kv[1])
    top_prob = top_count / shots

    print(f"Quantum result: top measured bitstring = {top_bitstring!r} "
          f"with probability {top_prob:.4f} over {shots} shots")
    print(f"Full counts: {counts}")

    passed = (top_bitstring == expected) and (top_prob > 0.9)

    print()
    if passed:
        print("PASS")
        return 0
    else:
        print("FAIL")
        return 1


if __name__ == "__main__":
    sys.exit(main())
