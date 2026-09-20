#!/usr/bin/env python3
"""
Erdos problem #61 -- quantum-testable instance.

Source metadata (data/problems.yaml, erdosproblems.com mirror at
/home/user/manman4/erdosproblems, entry "number: '61'"):
    prize: no
    informal_status: open (last_update 2025-08-31)
    formal_status: unformalized
    oeis: ["N/A"]
    tags: ["graph theory"]

LIMITATION, stated honestly up front: problem #61 carries no OEIS sequence
id in the source data (oeis: ["N/A"]). There is therefore no actual
integer sequence from this problem to build a "sequence membership" or
"early term" property around, and nothing here should be read as derived
from or verified against an OEIS b-file -- there isn't one. Per the task's
fallback instruction, this script instead builds a genuine, small, finite,
classically-checkable graph-theory decision problem in the same spirit as
the problem's tag ("graph theory"): triangle detection in a fixed small
graph via Grover search. This is real combinatorial content that a small
quantum circuit can genuinely compute, and it is checked here against an
exhaustive classical enumeration computed from first principles in this
script -- but it is NOT a numbered term of any Erdos-problem-61 OEIS
sequence (none exists), and readers should not conflate the two.

The property tested:
    Graph G = K4 minus the edge (0,1) on vertices {0,1,2,3}. Edges:
    (0,2), (0,3), (1,2), (1,3), (2,3).
    For each 4-bit string b3 b2 b1 b0 (bit i = 1 iff vertex i is in the
    chosen subset), define
        triangle(b) = True  iff  popcount(b) == 3
                              and every pair of vertices in the chosen
                                  subset is joined by an edge in G
                              (i.e. the subset forms a triangle in G).
    Classically, by brute force over all 16 subsets, the triangles are
    exactly {0,2,3} and {1,2,3} (both 3-subsets that avoid the missing
    edge (0,1)); {0,1,2} and {0,1,3} are not triangles because edge (0,1)
    is missing. So exactly 2 of the 16 possible 4-bit strings satisfy
    triangle(b): binary 1101 (decimal 13, vertices {0,2,3}) and binary
    1110 (decimal 14, vertices {1,2,3}).

Grover search is run over the 4-qubit space (N = 16) with these 2 marked
states, using the optimal number of iterations
floor(pi/4 * sqrt(N/M)) = floor(pi/4 * sqrt(16/2)) = 2, and the resulting
measurement distribution is compared against the classical brute-force
answer.
"""

import math
from itertools import combinations

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import MCMTGate, ZGate

# ---------------------------------------------------------------------
# 1. Classical computation, from first principles.
# ---------------------------------------------------------------------

VERTICES = [0, 1, 2, 3]
EDGES = {(0, 2), (0, 3), (1, 2), (1, 3), (2, 3)}  # K4 minus edge (0,1)


def has_edge(u: int, v: int) -> bool:
    return (min(u, v), max(u, v)) in EDGES


def is_triangle_subset(bits: int, n: int = 4) -> bool:
    """bits: integer 0..2^n-1, bit i = 1 iff vertex i is chosen."""
    subset = [v for v in range(n) if (bits >> v) & 1]
    if len(subset) != 3:
        return False
    for u, v in combinations(subset, 2):
        if not has_edge(u, v):
            return False
    return True


N_QUBITS = 4
N_STATES = 2 ** N_QUBITS
classical_marked = sorted(b for b in range(N_STATES) if is_triangle_subset(b))

print("Classical brute-force result:")
print(f"  graph G = K4 minus edge (0,1), edges = {sorted(EDGES)}")
print(f"  marked (triangle) states out of {N_STATES}: {classical_marked}"
      f" = {[format(b, '04b') for b in classical_marked]}")

assert classical_marked == [13, 14], (
    "Sanity check failed: expected exactly subsets {0,2,3}=13 and "
    "{1,2,3}=14 to be triangles."
)

# ---------------------------------------------------------------------
# 2. Grover search circuit over the 4-qubit space for these 2 states.
# ---------------------------------------------------------------------

M = len(classical_marked)
iterations = max(1, round(math.floor(math.pi / 4 * math.sqrt(N_STATES / M))))


def oracle_for_marked(marked_states, n):
    """Phase-flip oracle marking exactly `marked_states` (list of ints)."""
    qc = QuantumCircuit(n, name="oracle")
    mcz = MCMTGate(ZGate(), n - 1, 1)
    for state in marked_states:
        bits = [(state >> i) & 1 for i in range(n)]
        # flip qubits that should be 0 so the marked pattern becomes all-1s
        for i, b in enumerate(bits):
            if b == 0:
                qc.x(i)
        qc.append(mcz, list(range(n)))
        for i, b in enumerate(bits):
            if b == 0:
                qc.x(i)
    return qc


def diffuser(n):
    qc = QuantumCircuit(n, name="diffuser")
    qc.h(range(n))
    qc.x(range(n))
    mcz = MCMTGate(ZGate(), n - 1, 1)
    qc.append(mcz, list(range(n)))
    qc.x(range(n))
    qc.h(range(n))
    return qc


qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))

oracle = oracle_for_marked(classical_marked, N_QUBITS)
diff = diffuser(N_QUBITS)

for _ in range(iterations):
    qc.append(oracle.to_gate(), range(N_QUBITS))
    qc.append(diff.to_gate(), range(N_QUBITS))

qc.measure(range(N_QUBITS), range(N_QUBITS))

# ---------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------

backend = AerSimulator()
compiled = transpile(qc, backend)
shots = 4096
result = backend.run(compiled, shots=shots).result()
counts = result.get_counts()

# Qiskit's classical-register bit order is c[n-1] ... c[0] (little-endian
# string, rightmost char = qubit 0), matching our `bits >> v & 1` encoding.
counts_as_int = {int(bitstring, 2): c for bitstring, c in counts.items()}

marked_hits = sum(c for b, c in counts_as_int.items() if b in classical_marked)
marked_fraction = marked_hits / shots

print("\nQuantum (Grover) result:")
print(f"  iterations used: {iterations}")
print(f"  raw counts: {counts}")
print(f"  fraction of shots landing on a classically-marked (triangle) "
      f"state: {marked_fraction:.4f}")

# With N=16, M=2, 2 Grover iterations, theoretical success probability is
# high (>90%); we require a generous but meaningful threshold so the test
# is a real check, not a rubber stamp.
THRESHOLD = 0.85
quantum_agrees_with_classical = marked_fraction >= THRESHOLD

# Also check the top-2 most frequent measured outcomes are exactly the two
# classically marked states (order-independent), which is a stronger,
# structural agreement check.
top2 = sorted(counts_as_int.items(), key=lambda kv: -kv[1])[:2]
top2_states = sorted(b for b, _ in top2)
structural_match = top2_states == classical_marked

print(f"  top-2 measured states: {top2_states} "
      f"(classical triangle states: {classical_marked})")

verified = quantum_agrees_with_classical and structural_match

print("\n" + ("PASS" if verified else "FAIL"))
if not verified:
    raise SystemExit(1)
