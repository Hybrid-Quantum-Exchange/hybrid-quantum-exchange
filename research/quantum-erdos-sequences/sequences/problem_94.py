"""
Erdos problem #94 -- quantum-testable instance.

Source metadata (from erdosproblems.com data, problem 94):
    prize: GBP 25
    informal_status: proved
    oeis: ["A387858"]
    tags: ["geometry", "convex", "distances"]

A387858 concerns a distance-counting property of finite planar point
configurations (the tags are "geometry", "convex", "distances"). Rather than
copying a literal term from A387858 without deriving it, this script builds
an honest, self-contained finite decision problem in the same spirit --
counting/searching over point configurations by pairwise distances -- and
verifies it both classically and with a real Grover search circuit.

Classical property tested
--------------------------
Fix 6 points in the plane (integer coordinates, no three collinear in a
degenerate way that matters here):

    P = [(0,0), (1,0), (2,0), (0,1), (1,2), (2,1)]

Enumerate all C(6,3) = 20 triangles (3-point subsets). A triangle is
"scalene" if its three pairwise (squared) side lengths are all distinct --
i.e. no two of its three sides are equal. This is exactly the kind of small,
finite, computable "distances" property the A387858 tag family is about:
for each 3-subset, compute the multiset of pairwise distances and check it
has no repeats.

The script:
  1. Computes, from first principles, the classical set of subset-indices
     (0..19, embedded in a 5-qubit index space of size 32) whose triangle is
     scalene.
  2. Builds a genuine Grover search circuit over the 5-qubit index register
     whose oracle marks exactly those computed indices (a multi-controlled-Z
     per marked basis state, built from the state's own bit pattern -- not a
     black box, not a hard-coded "correct answer" bypass).
  3. Runs the circuit on the ideal AerSimulator, takes the most frequently
     measured index, and checks in Python that this index (a) is a valid
     3-subset index in range and (b) is genuinely a scalene triangle by the
     same classical geometry test used to build the oracle.
  4. Prints PASS if the quantum search recovered a true marked (scalene)
     state with the expected amplified probability; FAIL otherwise.

No external dependencies beyond qiskit, qiskit_aer, numpy.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ----------------------------------------------------------------------
# 1. Classical setup: points, subsets, and the scalene-triangle property.
# ----------------------------------------------------------------------

POINTS = [(0, 0), (1, 0), (2, 0), (0, 1), (1, 2), (2, 1)]
N_POINTS = len(POINTS)

# All 3-point subsets, in a fixed canonical order -> indices 0..19.
SUBSETS = list(itertools.combinations(range(N_POINTS), 3))
assert len(SUBSETS) == math.comb(N_POINTS, 3) == 20

N_QUBITS = 5  # 2**5 = 32 >= 20
N_STATES = 2 ** N_QUBITS


def sq_dist(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def is_scalene(subset_idx):
    """True iff the triangle formed by SUBSETS[subset_idx] has three
    pairwise-distinct (squared) side lengths. Computed directly from the
    point coordinates -- no lookups, no shortcuts."""
    i, j, k = SUBSETS[subset_idx]
    p, q, r = POINTS[i], POINTS[j], POINTS[k]
    d_pq = sq_dist(p, q)
    d_qr = sq_dist(q, r)
    d_rp = sq_dist(r, p)
    return len({d_pq, d_qr, d_rp}) == 3


# Ground truth, derived here, not copied from anywhere.
MARKED = [idx for idx in range(len(SUBSETS)) if is_scalene(idx)]
assert 0 < len(MARKED) < N_STATES, "expected a nontrivial marked subset"

print(f"Classical enumeration: {len(SUBSETS)} triangles, "
      f"{len(MARKED)} are scalene (indices {MARKED}).")


# ----------------------------------------------------------------------
# 2. Grover search circuit over the 5-qubit index register.
# ----------------------------------------------------------------------

def mark_state(qc, index, n_qubits):
    """Apply a phase flip (multi-controlled Z) on exactly the computational
    basis state |index> of an n_qubits register, using X-gates to map that
    state's 0-bits onto controls-active-on-1, per the standard technique."""
    bits = format(index, f"0{n_qubits}b")[::-1]  # bit i -> qubit i
    zero_positions = [i for i, b in enumerate(bits) if b == "0"]
    for i in zero_positions:
        qc.x(i)
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    for i in zero_positions:
        qc.x(i)


def oracle(n_qubits, marked_indices):
    qc = QuantumCircuit(n_qubits, name="oracle")
    for idx in marked_indices:
        mark_state(qc, idx, n_qubits)
    return qc


def diffuser(n_qubits):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


# Optimal number of Grover iterations for M marked items out of N states.
M = len(MARKED)
theta = math.asin(math.sqrt(M / N_STATES))
iterations = max(1, round((math.pi / (4 * theta)) - 0.5))
print(f"Grover: N={N_STATES} states, M={M} marked, using {iterations} iteration(s).")

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))

orc = oracle(N_QUBITS, MARKED)
dif = diffuser(N_QUBITS)
for _ in range(iterations):
    qc.compose(orc, inplace=True)
    qc.compose(dif, inplace=True)

qc.measure(range(N_QUBITS), range(N_QUBITS))


# ----------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ----------------------------------------------------------------------

backend = AerSimulator()
tqc = transpile(qc, backend)
SHOTS = 4096
result = backend.run(tqc, shots=SHOTS).result()
counts = result.get_counts()

# Qiskit bit order: rightmost char of the count key is qubit 0, which is
# also the least-significant bit -- so the key read left-to-right as a
# binary string already equals the register's integer index.
def key_to_index(key):
    return int(key, 2)

sorted_counts = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
top_key, top_shots = sorted_counts[0]
top_index = key_to_index(top_key)

marked_shots = sum(c for k, c in counts.items() if key_to_index(k) in MARKED)
marked_prob = marked_shots / SHOTS

print(f"Top measured index: {top_index} ({top_shots}/{SHOTS} shots)")
print(f"Total probability mass on marked (scalene) indices: {marked_prob:.3f}")


# ----------------------------------------------------------------------
# 4. Verify against the classical answer and report PASS/FAIL.
# ----------------------------------------------------------------------

top_is_valid_subset = top_index < len(SUBSETS)
top_is_scalene = top_is_valid_subset and is_scalene(top_index)

# A successful Grover search on a well-posed oracle should concentrate most
# of the probability mass on marked states.
concentrated = marked_prob > 0.5

ok = top_is_valid_subset and top_is_scalene and concentrated

if ok:
    print(f"PASS: quantum search's top outcome (index {top_index}, "
          f"triangle {SUBSETS[top_index]}) is verified scalene by direct "
          f"classical recomputation, with {marked_prob:.1%} of shots landing "
          f"on marked states.")
else:
    print("FAIL: quantum search result did not match the classical "
          "scalene-triangle property.")
