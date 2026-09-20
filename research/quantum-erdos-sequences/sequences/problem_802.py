"""
Erdos problem #802 -- quantum-testable lane.

Erdos problem #802 (see https://www.erdosproblems.com/802) is tagged only
"graph theory" in the erdosproblems.com dataset and its `oeis` field is
`["N/A"]` -- there is no OEIS sequence attached to this problem as of the
data snapshot checked (data/problems.yaml, entry `number: "802"`,
`informal_status.state: "open"`, `formal_status.state: "unformalized"`).

LIMITATION (read before trusting the "PASS"):
Because there is no OEIS sequence to derive a property from, this script
does NOT test anything that is specifically problem #802's open conjecture
(that would require formalizing an open research problem, which is out of
scope). Instead, honoring the "graph theory" tag, it tests a small, finite,
genuinely-computable graph-theory decision property that is representative
of the kind of object Erdos problems in this tag are about:

    Property tested: for a fixed graph G on 4 vertices (a 4-cycle, i.e.
    vertices 0-1-2-3-0 connected in a cycle, with the two diagonals 0-2 and
    1-3 NOT present), does there exist an independent set of size 2 (i.e. a
    pair of vertices with no edge between them)? And if so, which pairs are
    the independent pairs?

This is:
  - small (4 vertices, C(4,2) = 6 candidate pairs, encoded in 3 qubits),
  - finite and exactly computable classically (done from scratch below,
    by brute-force enumeration of all vertex pairs against the edge list),
  - a real search problem amenable to Grover's algorithm: the "database"
    is the 6 candidate pairs, the "good" items are the independent
    (non-edge) pairs, and we build a genuine phase-oracle + diffuser
    Grover circuit that amplifies exactly those pairs, then verify by
    measurement that the diagonals {0,2} and {1,3} (the only independent
    pairs in a 4-cycle) are the dominant measured outcomes.

Classical answer (computed below, not copied from anywhere):
  Graph: 4-cycle on vertices {0,1,2,3}, edges = {(0,1),(1,2),(2,3),(3,0)}.
  All C(4,2) = 6 pairs: (0,1) (0,2) (0,3) (1,2) (1,3) (2,3)
  Independent (non-edge) pairs -> {(0,2), (1,3)}  [the two diagonals]

The script builds an exact Grover oracle marking those 2 states out of the
6 valid pair-indices (padded into 3 qubits / 8 basis states, with the 2
unused index values excluded from being "good" by construction), runs it
on the ideal AerSimulator, and checks that the two diagonal pairs are the
top two measured outcomes -- i.e. that Grover amplification actually found
the independent sets.
"""

from itertools import combinations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------
# 1. Classical ground truth, computed from first principles.
# ---------------------------------------------------------------------

VERTICES = [0, 1, 2, 3]
EDGES = {(0, 1), (1, 2), (2, 3), (0, 3)}  # 4-cycle 0-1-2-3-0


def normalize(pair):
    a, b = pair
    return (a, b) if a < b else (b, a)


ALL_PAIRS = [normalize(p) for p in combinations(VERTICES, 2)]  # 6 pairs
assert len(ALL_PAIRS) == 6

INDEPENDENT_PAIRS = [p for p in ALL_PAIRS if p not in EDGES]
print("All vertex pairs:", ALL_PAIRS)
print("Edges:", sorted(EDGES))
print("Classically-computed independent (non-edge) pairs:", INDEPENDENT_PAIRS)

# Expect exactly the two diagonals of the 4-cycle.
assert set(INDEPENDENT_PAIRS) == {(0, 2), (1, 3)}, "unexpected classical result"

# Index the 6 pairs 0..5 in the fixed order of ALL_PAIRS; this index is
# what the Grover circuit searches over, encoded in 3 qubits (0..7, with
# indices 6 and 7 unused/never marked).
PAIR_INDEX = {pair: i for i, pair in enumerate(ALL_PAIRS)}
MARKED_INDICES = sorted(PAIR_INDEX[p] for p in INDEPENDENT_PAIRS)
print("Marked (good) indices in 3-qubit search space:", MARKED_INDICES)


# ---------------------------------------------------------------------
# 2. Grover search circuit over the 3-qubit index space.
# ---------------------------------------------------------------------

N_QUBITS = 3
N_STATES = 2 ** N_QUBITS  # 8


def apply_multi_controlled_z_on_bitstring(qc: QuantumCircuit, bitstring: str):
    """Flip the phase of exactly the basis state |bitstring>.

    bitstring is little-endian-agnostic here: index i of the string
    corresponds to qubit i. We open-control on qubits whose bit is '0'
    by sandwiching X gates, then apply a multi-controlled Z.
    """
    zero_qubits = [i for i, b in enumerate(bitstring) if b == "0"]
    for q in zero_qubits:
        qc.x(q)
    # Multi-controlled Z on all N_QUBITS qubits (phase flip of |11...1>
    # after the X sandwiching, which is |bitstring> in the original basis).
    qc.h(N_QUBITS - 1)
    qc.mcx(list(range(N_QUBITS - 1)), N_QUBITS - 1)
    qc.h(N_QUBITS - 1)
    for q in zero_qubits:
        qc.x(q)


def oracle(qc: QuantumCircuit, marked_indices):
    for idx in marked_indices:
        bitstring = format(idx, f"0{N_QUBITS}b")
        apply_multi_controlled_z_on_bitstring(qc, bitstring)


def diffuser(qc: QuantumCircuit):
    qc.h(range(N_QUBITS))
    qc.x(range(N_QUBITS))
    qc.h(N_QUBITS - 1)
    qc.mcx(list(range(N_QUBITS - 1)), N_QUBITS - 1)
    qc.h(N_QUBITS - 1)
    qc.x(range(N_QUBITS))
    qc.h(range(N_QUBITS))


def build_grover_circuit(marked_indices, n_states, n_qubits, iterations):
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(iterations):
        oracle(qc, marked_indices)
        diffuser(qc)
    qc.measure(range(n_qubits), range(n_qubits))
    return qc


# Optimal number of Grover iterations for M marked items out of N states.
M = len(MARKED_INDICES)
N = N_STATES
theta = np.arcsin(np.sqrt(M / N))
iterations = max(1, round((np.pi / 4) / theta - 0.5))
print(f"M={M} marked out of N={N} states -> {iterations} Grover iteration(s)")

qc = build_grover_circuit(MARKED_INDICES, N_STATES, N_QUBITS, iterations)

# ---------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------

sim = AerSimulator()
shots = 4096
compiled = transpile(qc, sim)
result = sim.run(compiled, shots=shots).result()
counts = result.get_counts()
print("Measurement counts:", counts)

# Qiskit's classical-register bit order in count keys is qubit N-1 ... qubit 0
# (big-endian string, little-endian qubit indexing). Convert each key back to
# our pair index (which used qubit i = bit i of the index, i.e. little-endian).
def key_to_index(key: str) -> int:
    little_endian = key[::-1]  # key[0] was qubit N-1; reverse -> qubit 0 first
    return int(little_endian, 2)


index_counts = {}
for bitkey, c in counts.items():
    idx = key_to_index(bitkey)
    index_counts[idx] = index_counts.get(idx, 0) + c

sorted_indices = sorted(index_counts.items(), key=lambda kv: -kv[1])
print("Counts by pair index (most frequent first):", sorted_indices)

top_indices = {idx for idx, _ in sorted_indices[:M]}
top_pairs = {ALL_PAIRS[i] for i in top_indices}
print("Top-M measured pairs:", top_pairs)


# ---------------------------------------------------------------------
# 4. Compare quantum result to the classical answer.
# ---------------------------------------------------------------------

verified = top_pairs == set(INDEPENDENT_PAIRS)

if verified:
    print("PASS")
else:
    print("FAIL")
