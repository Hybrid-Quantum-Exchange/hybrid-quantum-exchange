"""
Erdos problem #207 -- quantum-testable instance.

Source metadata (data/problems.yaml in manman4/erdosproblems, entry "number: 207"):
    oeis: ["N/A"]
    tags: ["combinatorics", "hypergraphs"]

LIMITATION: problem #207 has no associated OEIS sequence id (oeis: ["N/A"]
in the source data), so there is no OEIS "term" to check membership of, and
no natural integer sequence to search inside. This script is therefore the
best honest quantum-testable instance of the problem's actual mathematical
content, not a disguised OEIS lookup.

Statement of problem #207 (erdosproblems.com/207, proved by Kwan, Sah,
Sawhney, Simkin): for every g >= 2, for n large enough with n = 1 or 3 (mod 6),
there is a Steiner triple system on n points (a 3-uniform hypergraph in which
every pair of points lies in exactly one triple) such that any collection of
j edges, for 2 <= j <= g, spans at least j + 3 vertices. Read for j = 2: any
two distinct edges of the hypergraph must together span at least 5 vertices,
i.e. no two edges may share 2 of their 3 vertices (sharing 2 points would
force a repeated pair, and also collapses the union to only 4 vertices).

CLASSICAL PROPERTY TESTED (small, finite, computable):
    Fix a concrete 3-uniform hypergraph H on 7 vertices with 6 edges
    (constructed below, NOT a full Steiner system -- deliberately built with
    one intersecting pair of edges to give the search a genuine, checkable
    "yes" answer). Over all C(6,2) = 15 unordered pairs of distinct edges,
    mark a pair as VIOLATING the g=2 case of problem #207's spanning
    condition iff the two edges share exactly 2 vertices (equivalently, span
    only 4 < 5 = j+3 vertices). The classical answer -- the exact set of
    violating pair-indices -- is computed here in Python from first
    principles (no lookup table).

QUANTUM CIRCUIT: a genuine Grover search over 4 qubits (2^4 = 16 basis
states, indices 0..14 are real edge-pairs, index 15 is unused padding and is
never marked) whose oracle marks exactly the classically-computed violating
pair-indices with a phase flip, followed by the standard Grover diffuser, run
on AerSimulator. The circuit performs the optimal number of Grover
iterations for the known number of marked items. The most-probable measured
computational basis states are decoded back to pair-indices and compared
against the classical violating set.

PASS iff the set of pair-indices Grover amplifies to high probability
exactly equals the classically-computed violating set.
"""

import itertools
import math

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector

# ---------------------------------------------------------------------------
# 1. Classical construction: a 3-uniform hypergraph on 7 vertices, 6 edges.
#    One pair (edges 0 and 2) deliberately shares 2 vertices, to give the
#    search space a real, checkable "yes" instance rather than an empty one.
# ---------------------------------------------------------------------------
edges = [
    frozenset({0, 1, 2}),  # e0
    frozenset({0, 3, 4}),  # e1
    frozenset({0, 1, 3}),  # e2  -- shares {0,1} with e0: violation
    frozenset({2, 4, 5}),  # e3
    frozenset({1, 4, 6}),  # e4
    frozenset({2, 3, 6}),  # e5
]
n_edges = len(edges)
assert n_edges == 6

# All C(6,2) = 15 unordered pairs of distinct edges, in a fixed order.
pairs = list(itertools.combinations(range(n_edges), 2))
n_pairs = len(pairs)
assert n_pairs == 15

# Classical ground truth, computed from first principles: a pair (i, j)
# violates problem #207's j=2 spanning condition iff |e_i ∩ e_j| == 2
# (equivalently |e_i ∪ e_j| == 4 < 5).
classical_violations = []
for idx, (i, j) in enumerate(pairs):
    inter = len(edges[i] & edges[j])
    union = len(edges[i] | edges[j])
    if inter == 2:
        assert union == 4
        classical_violations.append(idx)
    else:
        # distinct 3-edges intersecting in 0 or 1 points span 6 or 5 >= 5
        assert union >= 5

classical_violations = sorted(classical_violations)
print("Edges:", [sorted(e) for e in edges])
print("Pairs (index -> (edge_i, edge_j)):")
for idx, (i, j) in enumerate(pairs):
    print(f"  {idx:2d}: e{i}={sorted(edges[i])} e{j}={sorted(edges[j])}")
print("Classical violating pair-indices (span < 5 vertices):", classical_violations)

# ---------------------------------------------------------------------------
# 2. Quantum circuit: Grover search over the 4-qubit index register (16
#    states; indices 15 unused/padding, never marked) for the violating
#    pair-indices computed above.
# ---------------------------------------------------------------------------
n_qubits = 4
N = 2 ** n_qubits  # 16
M = len(classical_violations)
assert M >= 1, "construction must yield at least one violation to search for"


def bits_of(k: int, n: int):
    """Little-endian bit list of k over n bits (qubit 0 = least significant)."""
    return [(k >> b) & 1 for b in range(n)]


def apply_marking(qc: QuantumCircuit, qubits, index: int):
    """Flip the sign of |index> via X-sandwiched multi-controlled Z."""
    bits = bits_of(index, len(qubits))
    for q, b in zip(qubits, bits):
        if b == 0:
            qc.x(q)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    for q, b in zip(qubits, bits):
        if b == 0:
            qc.x(q)


def oracle(qc: QuantumCircuit, qubits, marked_indices):
    for idx in marked_indices:
        apply_marking(qc, qubits, idx)


def diffuser(qc: QuantumCircuit, qubits):
    qc.h(qubits)
    qc.x(qubits)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    qc.x(qubits)
    qc.h(qubits)


qc = QuantumCircuit(n_qubits, n_qubits)
qubits = list(range(n_qubits))
qc.h(qubits)

# Optimal number of Grover iterations for N states, M marked items.
iterations = max(1, round((math.pi / 4) * math.sqrt(N / M)))
print(f"N={N} states, M={M} marked, Grover iterations={iterations}")

for _ in range(iterations):
    oracle(qc, qubits, classical_violations)
    diffuser(qc, qubits)

qc.measure(qubits, qubits)

# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator and compare to the classical answer.
# ---------------------------------------------------------------------------
sim = AerSimulator()
shots = 20000
result = sim.run(qc, shots=shots).result()
counts = result.get_counts()

# Decode bitstrings back to pair-indices (verified empirically to match the
# little-endian qubit-index convention used by the oracle/diffuser above).
decoded_counts = {}
for bitstring, c in counts.items():
    idx = int(bitstring, 2)
    decoded_counts[idx] = decoded_counts.get(idx, 0) + c

# Take the top-M most frequently measured indices -- these are what Grover
# amplified.
top_m = sorted(decoded_counts.items(), key=lambda kv: -kv[1])[:M]
measured_marked = sorted(idx for idx, _ in top_m)

print("Measured counts (top states):",
      sorted(decoded_counts.items(), key=lambda kv: -kv[1])[:M + 3])
print("Quantum-amplified top-M pair-indices:", measured_marked)
print("Classical violating pair-indices:     ", classical_violations)

# Sanity cross-check against the exact statevector (noiseless, no sampling
# noise) so a PASS is not an artifact of shot statistics alone.
qc_sv = QuantumCircuit(n_qubits)
qc_sv.h(qubits)
for _ in range(iterations):
    oracle(qc_sv, qubits, classical_violations)
    diffuser(qc_sv, qubits)
sv = Statevector(qc_sv)
probs = sv.probabilities(qubits)
sv_top_m = sorted(range(N), key=lambda k: -probs[k])[:M]
sv_top_m_sorted = sorted(sv_top_m)
total_marked_prob = sum(probs[i] for i in classical_violations)

print("Statevector top-M pair-indices:       ", sv_top_m_sorted)
print(f"Statevector total probability on marked set: {total_marked_prob:.4f}")

verified = (
    measured_marked == classical_violations
    and sv_top_m_sorted == classical_violations
    and total_marked_prob > 0.9
)

print("PASS" if verified else "FAIL")
assert verified, "quantum search result did not match classical violating set"
