#!/usr/bin/env python3
"""
Erdos problem #575 — quantum-testable instance.

Source metadata (erdosproblems.com, via data/problems.yaml in the manman4
clone of the repository): number "575", tags ["graph theory", "turan number"],
oeis: ["N/A"], status "disproved". There is NO OEIS sequence attached to this
problem (oeis field is literally the string "N/A"), so this script cannot be
built around "membership of an integer in an OEIS sequence" the way most
lanes in this library are. This is disclosed honestly rather than faking an
OEIS-based property.

LIMITATION: per the task's fallback instructions, this is a best-honest
attempt anchored in the problem's *tags* rather than an OEIS id. Problem 575
is a Turan-number question (extremal graph theory: the maximum number of
edges a graph on n vertices can have while avoiding a fixed forbidden
subgraph). To stay genuinely computable and quantum-testable in a handful of
qubits, this script tests the smallest non-trivial instance of the classical
Turan number that defines the field the problem lives in:

    ex(4, K3): the maximum number of edges a triangle-free (K3-free) graph on
    4 labeled vertices can have.

Classical fact (well known, and re-derived here from first principles, not
copied): K4 has 6 possible edges. By Turan/Mantel's theorem, the unique
extremal triangle-free graph on 4 vertices is the complete bipartite graph
K_{2,2} = the 4-cycle C4, with ex(4, K3) = 4 edges. The script enumerates all
2^6 = 64 edge subsets of K4 in Python, checks each for triangles by brute
force, and confirms this classically before using it as ground truth.

QUANTUM CIRCUIT: Grover search over the 6-qubit space of edge subsets of K4
(one qubit per potential edge), with an oracle built directly from the
classical brute-force enumeration above, marking exactly the edge-subsets
that are (a) triangle-free and (b) have exactly 4 edges (i.e. exactly the
Turan-extremal graphs). Grover's algorithm is run with the optimal number of
iterations for this marked-set size on AerSimulator, and the measurement
distribution is checked against the classical marked set: the circuit must
place (numerically, within a bounded tolerance) all of its measured weight
on classically-verified Turan-extremal 4-vertex triangle-free graphs, and
must recover the correct Turan number (max edge count among triangle-free
graphs) 4.

PASS/FAIL: the script prints PASS iff (1) the classically computed Turan
number ex(4,K3) equals 4, and (2) Grover search, run on the ideal
AerSimulator, returns (with overwhelming measured probability) only bit
strings that are independently verified classically to be triangle-free
4-edge subsets of K4.
"""

import itertools
import sys

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------------
# 1. Classical ground truth (first principles, no lookup).
# ---------------------------------------------------------------------------

VERTICES = 4
EDGES = list(itertools.combinations(range(VERTICES), 2))  # 6 possible edges
N_EDGES = len(EDGES)
assert N_EDGES == 6

ALL_TRIANGLES = list(itertools.combinations(range(VERTICES), 3))  # 4 triples


def edge_index(u, v):
    return EDGES.index((min(u, v), max(u, v)))


def has_triangle(edge_bits):
    """edge_bits: tuple of 0/1 of length N_EDGES, bit i == is EDGES[i] present."""
    present = set(EDGES[i] for i in range(N_EDGES) if edge_bits[i] == 1)
    for (a, b, c) in ALL_TRIANGLES:
        e1 = (min(a, b), max(a, b))
        e2 = (min(a, c), max(a, c))
        e3 = (min(b, c), max(b, c))
        if e1 in present and e2 in present and e3 in present:
            return True
    return False


def edge_count(edge_bits):
    return sum(edge_bits)


# Enumerate all 64 subsets of the 6 edges.
triangle_free_subsets = []
for bits in itertools.product([0, 1], repeat=N_EDGES):
    if not has_triangle(bits):
        triangle_free_subsets.append(bits)

turan_number = max(edge_count(b) for b in triangle_free_subsets)

extremal_subsets = [b for b in triangle_free_subsets if edge_count(b) == turan_number]

print(f"Classical enumeration: {len(triangle_free_subsets)} triangle-free "
      f"edge-subsets out of {2 ** N_EDGES} total.")
print(f"Classical Turan number ex(4, K3) = {turan_number}")
print(f"Number of extremal (Turan-optimal) triangle-free graphs on 4 "
      f"vertices: {len(extremal_subsets)}")

# Sanity check against the known closed-form answer (Mantel's theorem:
# ex(n, K3) = floor(n^2/4); for n=4 that's 4), derived independently above.
mantel_answer = (VERTICES * VERTICES) // 4
if turan_number != mantel_answer:
    print(f"FAIL: classical brute force ({turan_number}) disagrees with "
          f"Mantel's theorem closed form ({mantel_answer})")
    sys.exit(1)

# Bitstring convention used below: Qiskit reports measurement bitstrings with
# qubit 0 as the *rightmost* character. Build the marked set in that same
# convention up front so the oracle and the check agree.
def bits_to_qiskit_string(bits):
    # bits[i] is the value of qubit i; Qiskit prints c_{n-1}...c_1 c_0
    return "".join(str(bits[i]) for i in reversed(range(N_EDGES)))


marked_bitstrings = {bits_to_qiskit_string(b) for b in extremal_subsets}
print(f"Marked (target) bitstrings for Grover oracle: {sorted(marked_bitstrings)}")

# ---------------------------------------------------------------------------
# 2. Grover search over the 6-qubit space of edge subsets.
# ---------------------------------------------------------------------------

n = N_EDGES  # 6 qubits
marked_list = extremal_subsets  # list of tuples, qubit i == bits[i]


def apply_oracle(qc, qubits):
    """Phase-flip exactly the states in `marked_list` (multi-controlled Z,
    each preceded/followed by X gates on the qubits that must be 0)."""
    for bits in marked_list:
        zero_qubits = [qubits[i] for i in range(n) if bits[i] == 0]
        for q in zero_qubits:
            qc.x(q)
        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1])
        qc.h(qubits[-1])
        for q in zero_qubits:
            qc.x(q)


def apply_diffuser(qc, qubits):
    qc.h(qubits)
    qc.x(qubits)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    qc.x(qubits)
    qc.h(qubits)


N = 2 ** n
M = len(marked_list)
# Optimal number of Grover iterations for N states, M marked.
theta = np.arcsin(np.sqrt(M / N))
iterations = max(1, round((np.pi / (4 * theta)) - 0.5))
print(f"Grover: N={N} states, M={M} marked, running {iterations} iteration(s).")

qc = QuantumCircuit(n, n)
qc.h(range(n))
for _ in range(iterations):
    apply_oracle(qc, list(range(n)))
    apply_diffuser(qc, list(range(n)))
qc.measure(range(n), range(n))

sim = AerSimulator()
tqc = transpile(qc, sim)
shots = 4096
result = sim.run(tqc, shots=shots).result()
counts = result.get_counts()

# ---------------------------------------------------------------------------
# 3. Compare quantum result to the classical answer.
# ---------------------------------------------------------------------------

marked_shots = sum(c for bs, c in counts.items() if bs in marked_bitstrings)
marked_fraction = marked_shots / shots

top_bitstring, top_count = max(counts.items(), key=lambda kv: kv[1])
top_is_marked = top_bitstring in marked_bitstrings

print(f"Measured distribution over {len(counts)} distinct bitstrings.")
print(f"Fraction of shots landing on a classically-verified Turan-extremal "
      f"graph: {marked_fraction:.4f}")
print(f"Most frequent measured bitstring: {top_bitstring} "
      f"({top_count}/{shots} shots), classically Turan-extremal: {top_is_marked}")

ok = (
    turan_number == 4
    and turan_number == mantel_answer
    and top_is_marked
    and marked_fraction > 0.90
)

if ok:
    print("PASS")
else:
    print("FAIL")
    sys.exit(1)
