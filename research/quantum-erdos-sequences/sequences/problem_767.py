"""
Erdos problem #767 (per erdosproblems.com / the manman4/erdosproblems data dump)
------------------------------------------------------------------------------

Source metadata (data/problems.yaml, entry "number: '767'"):
    prize: no
    status: proved (as of 2025-08-31)
    oeis: ["possible"]
    tags: ["graph theory", "turan number"]

LIMITATION, stated up front: the "oeis" field for problem 767 in the source
data is the literal string "possible" -- not a real OEIS sequence id. There
is no A-number to pull a term from, so there is no genuine OEIS sequence to
build a quantum-testable membership/search property out of for *this specific
problem entry*. Faking an OEIS id would violate the task's own instructions.

Best-effort honest substitute: the problem's tags ("graph theory",
"turan number") do point at real, well-defined, small, computable
mathematics -- the Turan-type extremal number for triangle-free graphs,
i.e. Mantel's theorem (the n=3 case of Turan's theorem, which the "turan
number" tag is named after). This is a legitimate finite/computable
property in the same mathematical family as the problem, even though it is
not literally "the OEIS sequence for problem 767" (no such sequence id
exists in the source data).

Concretely, for n = 4 vertices:
    ex(4, K3) = floor(4^2 / 4) = 4
is the maximum number of edges a triangle-free graph on 4 labeled vertices
can have (Mantel's theorem). This script:

  1. Enumerates all 2^6 = 64 labeled graphs on 4 vertices (6 possible edges:
     01,02,03,12,13,23), classically, from first principles -- for each graph
     it checks every one of the C(4,3)=4 triangles and counts edges.
  2. Computes classically the exact set of graphs that are simultaneously
     triangle-free AND have the maximum edge count (4) -- this reproduces
     Mantel's ex(4,K3)=4 by direct enumeration, not by lookup.
  3. Builds a genuine Grover search circuit (6 qubits = one per possible
     edge, spanning the full 64-element search space) whose oracle marks
     exactly that classically-computed solution set (a diagonal phase-flip
     oracle built from the marked indices -- the standard way to turn a
     classical predicate into a Grover oracle for a moderate search space).
  4. Runs the circuit on the ideal AerSimulator and checks that the most
     probable measured outcomes are exactly the classically-marked graphs.
  5. Prints PASS if the quantum search's top outcomes match the classical
     solution set, FAIL otherwise.

This is a real amplitude-amplification computation over a real 64-element
search space, genuinely verified against a classical brute-force ground
truth computed in this script -- but, to be clear, it is NOT a test of any
OEIS sequence tied to Erdos problem #767 itself, because the source data
contains no valid OEIS id for that problem.
"""

import itertools
import math
from collections import Counter

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import MCXGate, DiagonalGate
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------------
# Step 1-2: classical ground truth, computed from first principles.
# ---------------------------------------------------------------------------

N_VERTICES = 4
EDGES = list(itertools.combinations(range(N_VERTICES), 2))  # 6 possible edges
N_EDGES = len(EDGES)
assert N_EDGES == 6
N_STATES = 2 ** N_EDGES  # 64

TRIANGLES = list(itertools.combinations(range(N_VERTICES), 3))  # 4 triples


def edge_index(u, v):
    return EDGES.index(tuple(sorted((u, v))))


def is_triangle_free(bits):
    """bits: tuple of 0/1 length N_EDGES, bit i says whether EDGES[i] is present."""
    for (a, b, c) in TRIANGLES:
        if bits[edge_index(a, b)] and bits[edge_index(b, c)] and bits[edge_index(a, c)]:
            return False
    return True


def edge_count(bits):
    return sum(bits)


# Enumerate all 64 graphs; classically find the max edge count among
# triangle-free graphs (this must equal Mantel's ex(4,K3) = floor(16/4) = 4).
max_tf_edges = -1
tf_graphs_by_count = {}
for i in range(N_STATES):
    bits = tuple((i >> k) & 1 for k in range(N_EDGES))
    if is_triangle_free(bits):
        c = edge_count(bits)
        tf_graphs_by_count.setdefault(c, []).append(i)
        if c > max_tf_edges:
            max_tf_edges = c

expected_mantel = (N_VERTICES * N_VERTICES) // 4  # floor(n^2/4)
assert max_tf_edges == expected_mantel, (
    f"classical brute force found ex(4,K3)={max_tf_edges}, "
    f"expected Mantel bound {expected_mantel}"
)

MARKED = sorted(tf_graphs_by_count[max_tf_edges])
print(f"Classical result: ex({N_VERTICES},K3) = {max_tf_edges} "
      f"(Mantel bound floor(n^2/4) = {expected_mantel})")
print(f"Number of extremal (triangle-free, max-edge) graphs on 4 labeled "
      f"vertices: {len(MARKED)}")
print(f"Marked search-space indices (out of {N_STATES}): {MARKED}")

# ---------------------------------------------------------------------------
# Step 3: build a Grover search circuit whose oracle marks exactly MARKED.
# ---------------------------------------------------------------------------

n_qubits = N_EDGES  # 6


def build_diagonal_oracle(marked_indices, n):
    """Phase-flip oracle: applies -1 to each marked computational basis state.

    Built directly as a diagonal unitary (a real, standard way to realize a
    Grover oracle once the marked set is known classically) rather than as a
    bit-pattern comparator, to keep the circuit compact for this write-up.
    """
    diag = np.ones(2 ** n, dtype=complex)
    for idx in marked_indices:
        diag[idx] = -1.0
    qc = QuantumCircuit(n, name="Oracle")
    qc.append(DiagonalGate(list(diag)), list(range(n)))
    return qc


def build_diffuser(n):
    qc = QuantumCircuit(n, name="Diffuser")
    qc.h(range(n))
    qc.x(range(n))
    qc.h(n - 1)
    if n - 1 >= 1:
        qc.append(MCXGate(n - 1), list(range(n - 1)) + [n - 1])
    qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))
    return qc


oracle = build_diagonal_oracle(MARKED, n_qubits)
diffuser = build_diffuser(n_qubits)

num_marked = len(MARKED)
theta = math.asin(math.sqrt(num_marked / N_STATES))
iterations = max(1, round((math.pi / 4 - theta / 2) / theta)) if theta > 0 else 1
# standard optimal-iteration formula for Grover with known marked-count
iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

qc = QuantumCircuit(n_qubits, n_qubits)
qc.h(range(n_qubits))
for _ in range(iterations):
    qc.append(oracle.to_instruction(), range(n_qubits))
    qc.append(diffuser.to_instruction(), range(n_qubits))
qc.measure(range(n_qubits), range(n_qubits))

print(f"Grover iterations used: {iterations} (search space {N_STATES}, "
      f"marked count {num_marked})")

# ---------------------------------------------------------------------------
# Step 4: run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

backend = AerSimulator()
tqc = transpile(qc, backend)
SHOTS = 4096
job = backend.run(tqc, shots=SHOTS)
result = job.result()
counts = result.get_counts()

# Qiskit's classical-register bit order is little-endian in the count keys
# (rightmost char = qubit 0), matching how we assigned edge_index -> qubit i.
measured = Counter()
for bitstring, freq in counts.items():
    idx = int(bitstring[::-1], 2)  # reverse to put qubit 0 in the low bit
    measured[idx] += freq

top_measured = [idx for idx, _ in measured.most_common(num_marked)]

# ---------------------------------------------------------------------------
# Step 5: compare quantum result to the classical answer.
# ---------------------------------------------------------------------------

marked_set = set(MARKED)
top_set = set(top_measured)
overlap = len(marked_set & top_set)

# Also report the total probability mass landing on marked states.
marked_shots = sum(measured.get(idx, 0) for idx in MARKED)
marked_fraction = marked_shots / SHOTS

print(f"Top {num_marked} measured outcomes (by frequency): {sorted(top_measured)}")
print(f"Overlap with classical marked set: {overlap}/{num_marked}")
print(f"Fraction of shots landing on a classically-marked (extremal) graph: "
      f"{marked_fraction:.3f}")

# Success criteria: Grover should concentrate amplitude heavily on the
# marked set. Require the top-N measured states to exactly equal the
# classical marked set, and the marked-state shot fraction to be well above
# the uniform-random baseline (num_marked / N_STATES).
baseline = num_marked / N_STATES
ran_ok = True
verified = (top_set == marked_set) and (marked_fraction > 3 * baseline)

if verified:
    print("PASS")
else:
    print("FAIL")
