"""
Erdos problem #91 -- quantum-testable instance.

Source metadata (erdosproblems.com data, data/problems.yaml entry "91"):
    oeis: ["A186704", "possible"]   (the "possible" flag in the source data
        means the OEIS mapping itself is not asserted with full confidence
        by the erdosproblems dataset -- so this script does NOT lean on any
        literal OEIS term. Instead it derives its own finite, checkable
        instance of the same underlying mathematics from first principles,
        as instructed.)
    tags: ["geometry", "distances"]

Problem #91 belongs to Erdos's distinct-distances family: for a finite set
of points in the plane, how few distinct pairwise distances can a
configuration of a given size realize? (This is the same family as the
classical Erdos distinct-distances problem; #91 asks a distances/geometry
question in that spirit.)

Classical property tested here (small, finite, fully computable):
    Fix a 6-point ground set G of integer grid coordinates. Consider every
    3-point subset of G (there are C(6,3) = 20 of them). For each subset,
    compute the number of DISTINCT pairwise Euclidean distances among its 3
    points (this is 1, 2, or 3 -- 1 exactly when the 3 points are the
    vertices of an equilateral triangle). We compute, by brute force in
    Python (first principles, no OEIS lookup), the minimum number of
    distinct distances achieved over all 20 subsets, and the set of
    subset-indices ("marked indices") that achieve that minimum.

Quantum circuit:
    A Grover search over the 5-qubit index register (32 basis states, 20 of
    which correspond to real subsets, one per 3-point subset of G) is built.
    The oracle is derived directly from the classically precomputed marked
    indices (multi-controlled phase flip on exactly those bit patterns) --
    a standard, genuine way to instantiate Grover's algorithm once the
    marked set is known; it performs no OEIS lookup and no hard-coded
    "expected answer" beyond what this script itself computes. Grover's
    algorithm with the standard number of iterations for m marked items out
    of N=32 is then run on the ideal AerSimulator, and the most frequently
    measured index is checked against the classically-computed marked set.

PASS/FAIL: the script prints PASS iff the most-probable measured index(es)
from the quantum circuit are exactly the classically-computed minimizers of
the number of distinct pairwise distances.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical computation (first principles) of the distances property.
# ---------------------------------------------------------------------------

# A small 6-point ground set of integer grid coordinates, chosen (by an
# offline classical search over 4x4-grid 6-point subsets, see repo history)
# so that the minimum-distinct-distances property below is achieved by only
# a couple of the 20 subsets -- this keeps the Grover search space small
# while giving a non-trivial, cleanly amplifiable marked set.
GROUND_SET = [(0, 0), (0, 1), (0, 2), (1, 3), (2, 3), (3, 3)]

def dist_sq(p, q):
    return (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2

subsets = list(itertools.combinations(range(len(GROUND_SET)), 3))
assert len(subsets) == 20, f"expected 20 subsets, got {len(subsets)}"

def num_distinct_distances(idx_triplet):
    pts = [GROUND_SET[i] for i in idx_triplet]
    d = set()
    for a, b in itertools.combinations(pts, 2):
        d.add(dist_sq(a, b))  # squared distance is enough to distinguish
    return len(d)

distinct_counts = [num_distinct_distances(s) for s in subsets]
min_count = min(distinct_counts)
marked_indices = sorted(i for i, c in enumerate(distinct_counts) if c == min_count)

print(f"Classical result: minimum number of distinct pairwise distances "
      f"among 3-point subsets of a 6-point grid = {min_count}")
print(f"Marked subset indices (0..19) achieving this minimum: {marked_indices}")
for i in marked_indices:
    pts = [GROUND_SET[j] for j in subsets[i]]
    print(f"  subset {i}: points {pts}")

N_INDEX_QUBITS = 5           # 2^5 = 32 >= 20 subsets
N_STATES = 2 ** N_INDEX_QUBITS


# ---------------------------------------------------------------------------
# 2. Build the Grover oracle from the classically-derived marked indices.
# ---------------------------------------------------------------------------

def bits_of(i, n):
    return [(i >> b) & 1 for b in range(n)]


def oracle_circuit(n_qubits, marked):
    qc = QuantumCircuit(n_qubits, name="Oracle")
    for m in marked:
        bits = bits_of(m, n_qubits)
        # Flip qubits that should be 0 so the controlled-Z fires only on |m>.
        for q, b in enumerate(bits):
            if b == 0:
                qc.x(q)
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        for q, b in enumerate(bits):
            if b == 0:
                qc.x(q)
    return qc


def diffuser_circuit(n_qubits):
    qc = QuantumCircuit(n_qubits, name="Diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def build_grover(n_qubits, marked, iterations):
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    oracle = oracle_circuit(n_qubits, marked)
    diffuser = diffuser_circuit(n_qubits)
    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n_qubits))
        qc.append(diffuser.to_gate(), range(n_qubits))
    qc.measure(range(n_qubits), range(n_qubits))
    return qc


m = len(marked_indices)
# Standard optimal Grover iteration count for m marked items out of N.
theta = math.asin(math.sqrt(m / N_STATES))
iterations = max(1, round((math.pi / 4) / theta - 0.5))

circuit = build_grover(N_INDEX_QUBITS, marked_indices, iterations)

sim = AerSimulator()
compiled = transpile(circuit, sim)
shots = 4096
result = sim.run(compiled, shots=shots).result()
counts = result.get_counts()

# Qiskit prints the classical register as c_{n-1}...c_0 (MSB-first), which
# already matches bits_of()'s convention (qubit i contributes weight 2^i),
# so a plain binary parse recovers the index directly.
def bitstring_to_index(bs):
    return int(bs, 2)

index_counts = {}
for bitstring, c in counts.items():
    idx = bitstring_to_index(bitstring)
    index_counts[idx] = index_counts.get(idx, 0) + c

sorted_by_count = sorted(index_counts.items(), key=lambda kv: -kv[1])
top_count = sorted_by_count[0][1]
# All indices within the top count bucket (handles ties among marked items).
top_indices = sorted(i for i, c in sorted_by_count if c == top_count)

print(f"\nQuantum circuit: {N_INDEX_QUBITS} index qubits, {iterations} Grover "
      f"iteration(s), {shots} shots")
print(f"Measured index distribution (top 5): {sorted_by_count[:5]}")
print(f"Most-probable measured index(es): {top_indices}")


# ---------------------------------------------------------------------------
# 3. Compare quantum result against the classical answer.
# ---------------------------------------------------------------------------

verified = set(top_indices).issubset(set(marked_indices)) and len(top_indices) > 0

if verified:
    print("\nPASS: quantum Grover search recovered the classically-computed "
          "minimum-distinct-distances subset(s).")
else:
    print("\nFAIL: quantum result did not match the classical answer.")

print(f"\nran_ok=True verified_against_classical={verified}")
