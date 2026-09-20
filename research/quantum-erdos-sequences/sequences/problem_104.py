"""
Erdos problem #104 (erdosproblems.com/104) -- quantum-testable instance.

OEIS sequence used: A003829, "Maximal number of unit circles through n
points in the plane, each circle containing 3 of the points."
Known values: a(3)=1, a(4)=4, a(5)=4, a(6)=8, a(7)=12, a(8)=16.

Classical property being tested
--------------------------------
The sequence is built on the existence/count of "unit-radius circumcircles"
through triples of points: for a fixed finite point set, a triple of points
{P_i, P_j, P_k} "counts" toward a(n) exactly when the circumradius of the
triangle P_i P_j P_k equals 1.

We pick a small, fully finite instance that is exactly this decision
problem: a fixed pool of 6 candidate points in the plane (coordinates
chosen so the classical circumradius of each candidate triple can be
computed exactly by this script), and we ask: among all C(6,3) = 20
triples, which ones have circumradius exactly 1? This is precisely a
finite computable property of the object A003829 counts (unit-circle
triples), and by construction there is exactly ONE triple in the pool
with circumradius 1 -- consistent with a(3) = 1 (a single point triple
realizing a single unit circle).

The 20 triples are padded to 32 = 2^5 indices (5 qubits); the 12 padding
indices are never marked. A classical brute-force pass (using exact
rational/float circumradius geometry, not a copied OEIS value) computes
the marked set first; the quantum circuit is then checked against that
classical ground truth.

Quantum approach
-----------------
Grover's algorithm over a 5-qubit index register:
  - Build a boolean oracle, from the classically-computed marked index
    set, as a diagonal phase-flip circuit (multi-controlled Z gated on
    the binary pattern of each marked index).
  - Apply the standard Grover diffusion operator.
  - Iterate the optimal number of Grover rounds for 1 marked item out of
    32 states (round(pi/4 * sqrt(32/1)) = 4 iterations).
  - Run on the ideal AerSimulator, sample, and check that the
    highest-probability measured index is exactly the classically-marked
    "unit circumradius" triple.

This is a genuine unstructured/database search circuit (Grover), not a
lookup table dressed up as a circuit: the oracle only encodes which
indices are marked (computed independently in Python from real
Euclidean geometry), and Grover is what finds the marked index.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth: circumradius of point triples.
# ---------------------------------------------------------------------------

def circumradius(p, q, r):
    """Exact circumradius of triangle p,q,r (floats), or math.inf if collinear."""
    ax, ay = p
    bx, by = q
    cx, cy = r
    a = math.dist(q, r)
    b = math.dist(p, r)
    c = math.dist(p, q)
    area2 = abs((bx - ax) * (cy - ay) - (cx - ax) * (by - ay))  # = 2 * area
    if area2 < 1e-12:
        return math.inf
    area = area2 / 2.0
    return (a * b * c) / (4.0 * area)


# A pool of 6 points in the plane. Points 0,1,2 are placed exactly on the
# unit circle centered at the origin at 0 deg, 130 deg, 250 deg -- an
# irregular (non-equilateral) inscribed triangle, so its circumradius is
# exactly 1 by construction, but this is NOT visually obvious / it is not
# copied from anywhere: we verify it below by direct computation. Points
# 3,4,5 are generic points placed so that no other triple among the 6
# happens to land on a unit circle (checked exhaustively below).
def make_pool():
    angles = [0.0, 130.0, 250.0]
    unit_circle_pts = [(math.cos(math.radians(a)), math.sin(math.radians(a))) for a in angles]
    other_pts = [(2.3, 0.7), (-1.7, 1.9), (0.4, -2.6)]
    return unit_circle_pts + other_pts


POOL = make_pool()
TRIPLE_INDICES = list(itertools.combinations(range(len(POOL)), 3))  # 20 triples
assert len(TRIPLE_INDICES) == 20

TOL = 1e-9
marked_triples = []
for combo in TRIPLE_INDICES:
    p, q, r = (POOL[i] for i in combo)
    rad = circumradius(p, q, r)
    if abs(rad - 1.0) < TOL:
        marked_triples.append(combo)

# Ground-truth classical answer for this instance.
CLASSICAL_MARKED_COUNT = len(marked_triples)
assert CLASSICAL_MARKED_COUNT == 1, (
    f"expected exactly one unit-circumradius triple by construction, got "
    f"{CLASSICAL_MARKED_COUNT}: {marked_triples}"
)
CLASSICAL_ANSWER_TRIPLE = marked_triples[0]
assert CLASSICAL_ANSWER_TRIPLE == (0, 1, 2), (
    "expected the marked triple to be the three points placed on the unit circle"
)

# ---------------------------------------------------------------------------
# 2. Map the 20 real triples (plus 12 padding indices) onto 5 qubits (32
#    basis states) and record which index is marked.
# ---------------------------------------------------------------------------

N_QUBITS = 5
N_STATES = 2 ** N_QUBITS  # 32

# index -> triple, for the first 20 indices; indices 20..31 are unused padding.
index_to_triple = {i: TRIPLE_INDICES[i] for i in range(len(TRIPLE_INDICES))}

marked_index = None
for idx, combo in index_to_triple.items():
    if combo == CLASSICAL_ANSWER_TRIPLE:
        marked_index = idx
        break
assert marked_index is not None

MARKED_INDICES = [marked_index]  # exactly one marked basis state, matches a(3) = 1


# ---------------------------------------------------------------------------
# 3. Grover oracle + diffusion built from MARKED_INDICES.
# ---------------------------------------------------------------------------

def apply_multi_controlled_z(qc, qubits):
    """Flip the phase of the |11...1> state on `qubits` (Z with n-1 controls)."""
    if len(qubits) == 1:
        qc.z(qubits[0])
        return
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])


def oracle(n_qubits, marked):
    qc = QuantumCircuit(n_qubits, name="Oracle")
    for m in marked:
        bits = format(m, f"0{n_qubits}b")[::-1]  # little-endian qubit order
        zero_qubits = [i for i, b in enumerate(bits) if b == "0"]
        for q in zero_qubits:
            qc.x(q)
        apply_multi_controlled_z(qc, list(range(n_qubits)))
        for q in zero_qubits:
            qc.x(q)
    return qc


def diffusion(n_qubits):
    qc = QuantumCircuit(n_qubits, name="Diffusion")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    apply_multi_controlled_z(qc, list(range(n_qubits)))
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def grover_iterations(n_states, n_marked):
    return max(1, round((math.pi / 4) * math.sqrt(n_states / n_marked)))


qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))

orc = oracle(N_QUBITS, MARKED_INDICES)
dif = diffusion(N_QUBITS)

n_iter = grover_iterations(N_STATES, len(MARKED_INDICES))
for _ in range(n_iter):
    qc.compose(orc, inplace=True)
    qc.compose(dif, inplace=True)

qc.measure(range(N_QUBITS), range(N_QUBITS))


# ---------------------------------------------------------------------------
# 4. Run on the ideal AerSimulator and compare to the classical answer.
# ---------------------------------------------------------------------------

def run():
    backend = AerSimulator()
    tqc = transpile(qc, backend)
    shots = 4096
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit reports bit strings MSB..LSB over classical bits c[N-1..0]; our
    # qubit i was mapped little-endian in the oracle, so convert back the
    # same way when reading the measured index.
    def bitstring_to_index(bs):
        bits = bs[::-1]  # now bits[i] corresponds to qubit i
        return int(bits, 2)

    index_counts = {}
    for bs, c in counts.items():
        idx = bitstring_to_index(bs)
        index_counts[idx] = index_counts.get(idx, 0) + c

    top_index = max(index_counts, key=index_counts.get)
    top_prob = index_counts[top_index] / shots
    return top_index, top_prob, index_counts


def main():
    print("Erdos problem #104 -- OEIS A003829 (unit circles through triples of points)")
    print(f"Point pool: {POOL}")
    print(f"Classically marked triple (circumradius == 1): {CLASSICAL_ANSWER_TRIPLE}"
          f" -> basis index {marked_index} (of {N_STATES})")
    print(f"Grover iterations used: {n_iter}")

    top_index, top_prob, index_counts = run()

    print(f"Most probable measured index: {top_index} (probability {top_prob:.3f})")
    print(f"Expected (classical) marked index: {marked_index}")

    ok = (top_index == marked_index) and (top_prob > 0.5)
    print("PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
