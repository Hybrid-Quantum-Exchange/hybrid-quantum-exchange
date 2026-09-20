#!/usr/bin/env python3
"""
Erdos problem #794 -- quantum-testable instance.

Source metadata (erdosproblems.com data, data/problems.yaml, entry "number: 794"):
    prize: no
    informal_status: disproved (2026-02-05)
    formal_status: Lean (2026-02-05)
    oeis: ["N/A"]
    tags: ["graph theory", "hypergraphs", "turan number"]

LIMITATION, stated honestly up front: problem #794 has NO associated OEIS
sequence id (the metadata literally records oeis: ["N/A"]). The task
instructions call for identifying a property from "its OEIS sequence id(s)
and tags" -- there is no id here, so nothing can be derived from an OEIS
sequence for this entry, and nothing below claims otherwise. What follows is
the best-effort fallback the instructions ask for in that case: a genuine,
finite, computable property taken directly from the problem's own subject
matter (Turan numbers / graph theory / hypergraphs, per its tags), verified
classically from first principles in this script, and then checked with a
real Grover-search quantum circuit. No OEIS value of any kind is used,
copied, or claimed.

Classical property being tested
--------------------------------
The Turan number ex(n, K_3): the maximum number of edges a triangle-free
graph on n vertices can have. For n = 4 (K_4 has 6 possible edges), this
script:

  1. Brute-forces, classically, over all 2^6 = 64 edge-subsets of K_4 to find
     the true maximum edge count of a triangle-free subgraph, and the exact
     set of edge-subsets ("colorings") that achieve it. This is the ground
     truth, computed independently of the quantum part.

  2. Builds a 6-qubit Grover search circuit (one qubit per edge of K_4) whose
     oracle marks exactly the bitstrings representing a triangle-free graph
     with that maximum edge count (the oracle is constructed as an exact
     multi-controlled phase flip on those specific classically-verified
     marked computational basis states -- this is the standard way to build
     a Grover oracle for a combinatorial predicate in a small instance).

  3. Runs the circuit on the ideal AerSimulator and checks that the
     overwhelming majority of measurement outcomes land on states that are
     independently re-verified (by the same classical triangle/edge-count
     check) to be maximum triangle-free colorings of K_4.

Edge indexing (K_4, vertices 0,1,2,3), qubit q_i <-> edge:
    q0=(0,1) q1=(0,2) q2=(0,3) q3=(1,2) q4=(1,3) q5=(2,3)

PASS/FAIL: PASS iff the classically-computed maximum is ex(4,K3) = 4 (the
well known Turan/Mantel value floor(4^2/4) = 4, achieved by K_{2,2}) AND the
quantum circuit's measurement distribution is concentrated (>= 90% of shots)
on states that satisfy the same triangle-free + max-edge-count predicate.
"""

import itertools
import sys

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np

# ---------------------------------------------------------------------------
# 1. Classical ground truth (first principles, no external data)
# ---------------------------------------------------------------------------

EDGES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]  # index -> edge
TRIANGLES = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]


def edge_index(u, v):
    a, b = min(u, v), max(u, v)
    return EDGES.index((a, b))


def is_triangle_free(bits):
    """bits: tuple of 6 ints (0/1), one per edge in EDGES order."""
    for (a, b, c) in TRIANGLES:
        i, j, k = edge_index(a, b), edge_index(a, c), edge_index(b, c)
        if bits[i] and bits[j] and bits[k]:
            return False
    return True


def classical_max_triangle_free_edges():
    best = 0
    for bits in itertools.product([0, 1], repeat=6):
        if is_triangle_free(bits) and sum(bits) > best:
            best = sum(bits)
    return best


def marked_bitstrings(max_edges):
    """All edge-subsets achieving the max triangle-free edge count."""
    marked = []
    for bits in itertools.product([0, 1], repeat=6):
        if sum(bits) == max_edges and is_triangle_free(bits):
            marked.append(bits)
    return marked


CLASSICAL_MAX = classical_max_triangle_free_edges()
MARKED = marked_bitstrings(CLASSICAL_MAX)

# Independent sanity check against the known closed form for Mantel's
# theorem (the n=3 case of Turan's theorem): ex(n, K_3) = floor(n^2 / 4).
EXPECTED_MANTEL = (4 * 4) // 4  # = 4
assert CLASSICAL_MAX == EXPECTED_MANTEL, (
    f"classical brute force ({CLASSICAL_MAX}) disagrees with Mantel's "
    f"formula ({EXPECTED_MANTEL})"
)
assert len(MARKED) > 0

print(f"Classical result: ex(4, K_3) = {CLASSICAL_MAX} "
      f"(Mantel/Turan closed form floor(4^2/4) = {EXPECTED_MANTEL})")
print(f"Number of maximum triangle-free edge-colorings of K_4: {len(MARKED)}")
for m in MARKED:
    edges_present = [EDGES[i] for i in range(6) if m[i]]
    print(f"  bits={m}  edges={edges_present}")

# ---------------------------------------------------------------------------
# 2. Quantum Grover-search circuit
# ---------------------------------------------------------------------------

N_QUBITS = 6


def apply_marking_oracle(qc, marked_states):
    """Phase-flip exactly the given computational basis states.

    For each marked bitstring, X-gate the 0-bits into 1, apply a
    multi-controlled Z (phase flip on |111111>), then undo the X-gates.
    This is an exact oracle for the classically-precomputed marked set --
    a standard construction for small Grover instances.
    """
    for bits in marked_states:
        zero_positions = [i for i, b in enumerate(bits) if b == 0]
        for i in zero_positions:
            qc.x(i)
        qc.h(N_QUBITS - 1)
        qc.mcx(list(range(N_QUBITS - 1)), N_QUBITS - 1)
        qc.h(N_QUBITS - 1)
        for i in zero_positions:
            qc.x(i)


def diffuser(qc):
    qc.h(range(N_QUBITS))
    qc.x(range(N_QUBITS))
    qc.h(N_QUBITS - 1)
    qc.mcx(list(range(N_QUBITS - 1)), N_QUBITS - 1)
    qc.h(N_QUBITS - 1)
    qc.x(range(N_QUBITS))
    qc.h(range(N_QUBITS))


def build_grover_circuit(marked_states, n_qubits=N_QUBITS):
    n_marked = len(marked_states)
    n_total = 2 ** n_qubits
    # Optimal number of Grover iterations for n_marked solutions out of
    # n_total, standard formula: floor(pi/4 * sqrt(N/M)).
    iterations = max(1, int(np.floor((np.pi / 4) * np.sqrt(n_total / n_marked))))

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(iterations):
        apply_marking_oracle(qc, marked_states)
        diffuser(qc)
    qc.measure(range(n_qubits), range(n_qubits))
    return qc, iterations


qc, iterations = build_grover_circuit(MARKED)
print(f"\nGrover circuit: {N_QUBITS} qubits, {iterations} iteration(s), "
      f"{len(MARKED)} marked states out of {2 ** N_QUBITS}")

# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator and check the result
# ---------------------------------------------------------------------------

sim = AerSimulator()
tqc = transpile(qc, sim)
SHOTS = 4096
result = sim.run(tqc, shots=SHOTS).result()
counts = result.get_counts()

marked_set = {tuple(int(c) for c in reversed(b)) for b in
              ("".join(str(x) for x in bits) for bits in MARKED)}
# Qiskit returns bitstrings with qubit 0 as the rightmost character;
# rebuild the marked set in that same convention for comparison.
marked_bitstrings_qiskit = set()
for bits in MARKED:
    # bits[i] is qubit i; qiskit string order is q_{n-1}...q_1 q_0
    s = "".join(str(bits[i]) for i in reversed(range(N_QUBITS)))
    marked_bitstrings_qiskit.add(s)

hits = sum(v for k, v in counts.items() if k in marked_bitstrings_qiskit)
frac_hits = hits / SHOTS

print(f"\nTop measured outcomes:")
for k, v in sorted(counts.items(), key=lambda kv: -kv[1])[:8]:
    tag = "MARKED" if k in marked_bitstrings_qiskit else ""
    print(f"  {k}: {v:5d}  {tag}")

print(f"\nFraction of shots landing on a verified maximum triangle-free "
      f"coloring: {frac_hits:.3f} ({hits}/{SHOTS})")

# Independently re-verify every hit bitstring against the classical
# predicate (not just membership in the precomputed set), as an extra
# consistency check that the quantum-measured answers really are correct
# triangle-free, max-edge-count graphs.
def bitstring_to_bits_tuple(s):
    # s is qiskit order q_{n-1}...q0; convert back to bits[i] = qubit i
    return tuple(int(s[N_QUBITS - 1 - i]) for i in range(N_QUBITS))

reverify_ok = True
for k in counts:
    if k in marked_bitstrings_qiskit:
        bits = bitstring_to_bits_tuple(k)
        if not (is_triangle_free(bits) and sum(bits) == CLASSICAL_MAX):
            reverify_ok = False

VERIFIED = (frac_hits >= 0.90) and reverify_ok and (CLASSICAL_MAX == EXPECTED_MANTEL)

print(f"\nRe-verification of all marked hits against classical predicate: "
      f"{'OK' if reverify_ok else 'FAILED'}")

if VERIFIED:
    print("\nPASS")
    sys.exit(0)
else:
    print("\nFAIL")
    sys.exit(1)
