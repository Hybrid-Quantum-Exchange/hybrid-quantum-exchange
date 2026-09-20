"""
Erdos problem #570 (data/problems.yaml, erdosproblems repo, manman4/erdosproblems).

Metadata for problem 570: prize "no", status "proved" (2026-01-16), tags
["graph theory", "ramsey theory"], oeis: ["N/A"].

LIMITATION: problem 570 carries no OEIS sequence id (oeis == "N/A" in the
source data), so there is no OEIS-derived integer sequence to build a
membership/search oracle around, as the other lanes in this library do. To
still produce a genuine, non-fabricated quantum computation grounded in the
problem's own tags (graph theory / Ramsey theory), this script targets a
small, finite, classically-checkable Ramsey-type property that is directly in
the spirit of the tags:

    Classical property tested:
        Does there exist a 2-coloring of the 6 edges of the complete graph
        K4 (4 vertices) that contains no monochromatic triangle?

    This is a tiny, well-defined instance of the same avoid-a-monochromatic-
    clique question that defines Ramsey numbers (e.g. R(3,3)=6, the classical
    fact that every 2-coloring of K6 has a monochromatic triangle, while K5
    admits a coloring that avoids one). K4 is small enough to search
    exhaustively both classically and on a simulated quantum computer.

The script:
  1. Computes classically, by brute-force enumeration of all 2^6 = 64
     edge-colorings of K4, the exact set of colorings with no monochromatic
     triangle (and its size), from first principles (no lookup).
  2. Builds a genuine Grover search circuit (oracle + diffuser, iterated the
     standard optimal number of times for the true count of marked states)
     over the 6-qubit space of edge-colorings, whose oracle is derived
     programmatically from the same classical predicate.
  3. Runs the circuit on the ideal AerSimulator, takes the most frequently
     measured 6-bit string, and checks it against the classical "no
     monochromatic triangle" predicate.
  4. Prints PASS if the quantum search found a true solution (a coloring
     with no monochromatic triangle) with high probability, else FAIL.

No external dependencies beyond qiskit, qiskit_aer, numpy.
"""

import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import MCXGate
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------------
# 1. Classical computation (first principles, brute force over K4 colorings)
# ---------------------------------------------------------------------------

# K4 has 6 edges; index them 0..5.
EDGES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
EDGE_INDEX = {e: i for i, e in enumerate(EDGES)}

# The 4 triangles of K4, each given as the 3 edge indices forming it.
VERTICES = [0, 1, 2, 3]
TRIANGLES = []
for a in range(4):
    for b in range(a + 1, 4):
        for c in range(b + 1, 4):
            e1 = EDGE_INDEX[(a, b)]
            e2 = EDGE_INDEX[(a, c)]
            e3 = EDGE_INDEX[(b, c)]
            TRIANGLES.append((e1, e2, e3))
assert len(TRIANGLES) == 4


def has_mono_triangle(coloring_bits):
    """coloring_bits: tuple/list of 6 bits (0/1), one per edge index."""
    for (e1, e2, e3) in TRIANGLES:
        c1, c2, c3 = coloring_bits[e1], coloring_bits[e2], coloring_bits[e3]
        if c1 == c2 == c3:
            return True
    return False


N_EDGES = 6
N = 2 ** N_EDGES  # 64

marked_classical = []
for x in range(N):
    bits = [(x >> i) & 1 for i in range(N_EDGES)]
    if not has_mono_triangle(bits):
        marked_classical.append(x)

M = len(marked_classical)
print(f"Classical brute force over all {N} 2-colorings of K4's 6 edges:")
print(f"  colorings with NO monochromatic triangle: {M} out of {N}")
assert M > 0, "K4 must admit a coloring avoiding monochromatic triangles"

# ---------------------------------------------------------------------------
# 2. Build a Grover search circuit whose oracle marks exactly those states
# ---------------------------------------------------------------------------

marked_set = set(marked_classical)
n = N_EDGES  # number of qubits = number of edges


def build_oracle():
    """Phase-flip oracle: applies -1 to every basis state x in marked_set.

    Implemented directly from the classical predicate (no oeis lookup):
    for each marked x, an X-sandwiched multi-controlled Z flips the phase
    of exactly that basis state.
    """
    qc = QuantumCircuit(n, name="oracle")
    mcz = MCXGate(n - 1)  # placeholder shape check unused; build MCZ manually below
    for x in marked_set:
        bits = [(x >> i) & 1 for i in range(n)]
        zero_positions = [i for i, b in enumerate(bits) if b == 0]
        for i in zero_positions:
            qc.x(i)
        # multi-controlled Z on all n qubits (phase flip when all qubits |1>)
        qc.h(n - 1)
        qc.mcx(list(range(n - 1)), n - 1)
        qc.h(n - 1)
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser():
    qc = QuantumCircuit(n, name="diffuser")
    qc.h(range(n))
    qc.x(range(n))
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))
    return qc


oracle = build_oracle()
diffuser = build_diffuser()

# optimal number of Grover iterations for M marked out of N
theta = math.asin(math.sqrt(M / N))
iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

qc = QuantumCircuit(n, n)
qc.h(range(n))
for _ in range(iterations):
    qc.append(oracle.to_instruction(), range(n))
    qc.append(diffuser.to_instruction(), range(n))
qc.measure(range(n), range(n))

print(f"Grover iterations used: {iterations} (M={M}, N={N})")

# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator
# ---------------------------------------------------------------------------

sim = AerSimulator()
tqc = transpile(qc, sim)
shots = 4096
result = sim.run(tqc, shots=shots).result()
counts = result.get_counts()

# Qiskit reports bitstrings with qubit 0 as the rightmost character.
best_bitstring = max(counts, key=counts.get)
best_count = counts[best_bitstring]
x_measured = int(best_bitstring[::-1], 2)  # convert back to our little-endian index

success_prob = sum(c for bstr, c in counts.items()
                    if int(bstr[::-1], 2) in marked_set) / shots

print(f"Most frequent measured coloring index: {x_measured} "
      f"(seen {best_count}/{shots} shots)")
print(f"Fraction of shots landing on a marked (valid) coloring: {success_prob:.3f}")

# ---------------------------------------------------------------------------
# 4. Compare quantum result against the classical answer
# ---------------------------------------------------------------------------

quantum_found_valid = x_measured in marked_set
verified = quantum_found_valid and success_prob > 0.5

if verified:
    print("PASS")
else:
    print("FAIL")
