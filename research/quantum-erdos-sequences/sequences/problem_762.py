"""
Erdos problem #762 (per data/problems.yaml: state "disproved (Lean)",
tags ["graph theory", "chromatic number"], oeis: ["N/A"]).

LIMITATION, stated honestly up front: problem #762 has NO associated OEIS
sequence id in the source data (oeis: ["N/A"]). There is therefore no
"sequence" to make quantum-testable in the sense the rest of this library
uses (membership/term-of-sequence checks against an OEIS id). This script
is the best-effort fallback described for that case: it builds a genuine,
correctness-checked quantum circuit around the one piece of real
mathematical content #762's tags do give us -- chromatic-number graph
coloring -- rather than fabricating or copying an OEIS value that does not
exist for this problem.

Chosen finite, computable property
-----------------------------------
Graph: the 4-cycle C4 (vertices 0,1,2,3; edges (0,1),(1,2),(2,3),(3,0)).
Property tested: "C4 has a proper 2-coloring" (equivalently, chromatic
number of C4 is <= 2; C4 is bipartite so this is true, and in fact
chi(C4) = 2 exactly since C4 contains an edge).

Search space: all 2^4 = 16 assignments of 2 colors to the 4 vertices,
encoded as 4 qubits (one qubit per vertex, |0>/|1> = color).

Classical ground truth (computed here by brute force, first principles):
enumerate all 16 colorings, keep those where every edge has differing
colors. This is done in Python with no reference to any external table.

Quantum method: Grover search. A marking oracle flags the "good" colorings
(no monochromatic edge) using per-edge XOR + multi-controlled-Z; the
diffusion operator amplifies them. With M valid colorings out of N=16
states, ceil(pi/4 * sqrt(N/M)) Grover iterations are applied (computed
from M, not hard-coded per instance), then the circuit is measured on the
ideal AerSimulator. PASS requires: (a) the classical brute force confirms
C4 is properly 2-colorable (M > 0), and (b) the most frequent measured
bitstring over many shots is one of the classically valid colorings.
"""

import itertools
import math

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------------
# Instance: C4 (4-cycle)
# ---------------------------------------------------------------------------
N_VERTICES = 4
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0)]
N = 2 ** N_VERTICES  # 16 total colorings


def is_proper_2_coloring(bits):
    """bits: tuple of 0/1, length N_VERTICES. True if no edge is monochromatic."""
    return all(bits[u] != bits[v] for u, v in EDGES)


def classical_valid_colorings():
    """Brute force over all 2^4 assignments, first principles."""
    valid = []
    for combo in itertools.product([0, 1], repeat=N_VERTICES):
        if is_proper_2_coloring(combo):
            valid.append(combo)
    return valid


VALID = classical_valid_colorings()
M = len(VALID)
CHROMATIC_NUMBER_LE_2 = M > 0

print("Classical brute-force result:")
print(f"  Total colorings checked: {N}")
print(f"  Proper 2-colorings found: {M} -> {VALID}")
print(f"  C4 properly 2-colorable (chi(C4) <= 2): {CHROMATIC_NUMBER_LE_2}")

if M == 0:
    # No solutions -- Grover has nothing to amplify toward. Report honestly.
    print("No valid colorings exist for this instance; a Grover search has "
          "no marked states to amplify. Declining to fake a circuit result.")
    RAN_OK = True
    VERIFIED = False
else:
    # -----------------------------------------------------------------------
    # Grover oracle: mark bitstrings (q0 q1 q2 q3) = colors of vertices 0..3
    # such that every edge (u,v) has differing colors, i.e. q_u XOR q_v = 1
    # for all 4 edges. We use one ancilla per edge to hold the XOR, and a
    # multi-controlled-Z on the 4 edge-ancillas to flip phase when ALL are 1.
    # -----------------------------------------------------------------------
    n_edges = len(EDGES)

    def build_oracle():
        qc = QuantumCircuit(N_VERTICES + n_edges, name="oracle")
        vertex_qubits = list(range(N_VERTICES))
        edge_qubits = list(range(N_VERTICES, N_VERTICES + n_edges))

        # compute edge XORs into ancillas
        for i, (u, v) in enumerate(EDGES):
            qc.cx(vertex_qubits[u], edge_qubits[i])
            qc.cx(vertex_qubits[v], edge_qubits[i])

        # phase flip when all edge ancillas are 1 (all edges properly colored)
        qc.h(edge_qubits[-1])
        qc.mcx(edge_qubits[:-1], edge_qubits[-1])
        qc.h(edge_qubits[-1])

        # uncompute edge XORs (restore ancillas to |0>)
        for i, (u, v) in enumerate(EDGES):
            qc.cx(vertex_qubits[v], edge_qubits[i])
            qc.cx(vertex_qubits[u], edge_qubits[i])
        return qc

    def build_diffuser(n):
        qc = QuantumCircuit(n, name="diffuser")
        qc.h(range(n))
        qc.x(range(n))
        qc.h(n - 1)
        qc.mcx(list(range(n - 1)), n - 1)
        qc.h(n - 1)
        qc.x(range(n))
        qc.h(range(n))
        return qc

    n_vertex_qubits = N_VERTICES
    n_total_qubits = N_VERTICES + n_edges

    oracle = build_oracle()
    diffuser = build_diffuser(n_vertex_qubits)

    iterations = max(1, round((math.pi / 4) * math.sqrt(N / M)))

    qc = QuantumCircuit(n_total_qubits, n_vertex_qubits)
    qc.h(range(n_vertex_qubits))  # equal superposition over all colorings

    for _ in range(iterations):
        qc.compose(oracle, qubits=range(n_total_qubits), inplace=True)
        qc.compose(diffuser, qubits=range(n_vertex_qubits), inplace=True)

    qc.measure(range(n_vertex_qubits), range(n_vertex_qubits))

    backend = AerSimulator()
    tqc = transpile(qc, backend)
    shots = 4096
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit bit order is little-endian in the classical register string;
    # register bit i was measured from vertex qubit i, and the printed
    # string has qubit (n-1) as the leftmost character.
    def bitstring_to_vertex_tuple(bstr):
        # bstr like 'q3q2q1q0'
        rev = bstr[::-1]
        return tuple(int(c) for c in rev)

    most_common_str = max(counts, key=counts.get)
    most_common_tuple = bitstring_to_vertex_tuple(most_common_str)
    most_common_freq = counts[most_common_str] / shots

    print("\nGrover search:")
    print(f"  Marked (valid) colorings M = {M}, search space N = {N}, "
          f"iterations = {iterations}")
    print(f"  Most frequent measured coloring: {most_common_tuple} "
          f"(bitstring '{most_common_str}', freq {most_common_freq:.3f})")
    print(f"  Classically valid colorings: {VALID}")

    RAN_OK = True
    VERIFIED = most_common_tuple in VALID

print("\nRESULT:", "PASS" if VERIFIED else "FAIL")
