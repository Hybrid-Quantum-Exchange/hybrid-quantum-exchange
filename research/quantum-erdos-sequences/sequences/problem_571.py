"""
Erdos problem #571 -- quantum-testable instance.

Source: /home/user/manman4/erdosproblems/data/problems.yaml, entry
`number: "571"`. That entry has:
    oeis: ["N/A"]
    tags: ["graph theory", "turan number"]
    informal_status: proved

LIMITATION (stated up front, honestly): problem #571 carries no OEIS
sequence id ("N/A"), so there is no OEIS-defined integer sequence to test
membership/terms of. Per the task's fallback instructions, this script
instead builds a genuine, small, finite, computable property that is
faithful to the problem's own tags ("graph theory", "turan number") --
Turan-type extremal graph counting -- and verifies it with a real Grover
search circuit. This is a best-honest-attempt substitute for a sequence
membership test, not a test of an actual OEIS sequence, because none
exists for this problem.

The property tested
--------------------
Turan's theorem for triangles (K_3): among all graphs on n=4 labeled
vertices, the maximum number of edges a TRIANGLE-FREE graph can have is

    ex(4, K_3) = floor(4^2 / 4) = 4

(achieved by the complete bipartite graph K_{2,2} and its relabelings).

K4 has C(4,2) = 6 possible edges, indexed e0..e5 in the fixed order
    e0=(0,1) e1=(0,2) e2=(0,3) e3=(1,2) e4=(1,3) e5=(2,3)
so every graph on 4 vertices is a 6-bit string (bit i = 1 iff edge i is
present). The four possible triangles correspond to these edge-index
triples:
    {0,1,2} -> (e0,e1,e3)
    {0,1,3} -> (e0,e2,e4)
    {0,2,3} -> (e1,e2,e5)
    {1,2,3} -> (e3,e4,e5)

The classical property under test: "graphs on 4 vertices with exactly
ex(4,K_3)=4 edges and no triangle" -- i.e. the maximizers of Turan's
theorem for n=4, r=3. The script first *derives* this set and its size
by brute-force enumeration over all 2^6 = 64 graphs (first principles,
no lookup), then builds a genuine Grover search circuit over 6 qubits
whose oracle marks exactly those bitstrings, and checks that running
the circuit on the ideal AerSimulator recovers that same marked set.

Classical answer for this instance (computed below, from scratch):
the marked set is exactly the 3 graphs isomorphic to K_{2,2} on 4
labeled vertices (the 3 ways to 2-color 4 vertices into two pairs),
each with exactly 4 edges and no triangle.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------
# 1. Classical derivation (first principles) of the property + answer
# ---------------------------------------------------------------------

N_VERTICES = 4
EDGES = list(itertools.combinations(range(N_VERTICES), 2))  # e0..e5
N_EDGES = len(EDGES)  # 6
assert N_EDGES == 6

TRIANGLES = list(itertools.combinations(range(N_VERTICES), 3))
# For each triangle {a,b,c}, the indices into EDGES of its 3 edges.
TRIANGLE_EDGE_IDX = []
for (a, b, c) in TRIANGLES:
    idx = tuple(sorted(EDGES.index(tuple(sorted(pair))) for pair in
                        [(a, b), (a, c), (b, c)]))
    TRIANGLE_EDGE_IDX.append(idx)


def is_triangle_free(bits):
    """bits: tuple of 0/1 of length N_EDGES (edge presence)."""
    for (i, j, k) in TRIANGLE_EDGE_IDX:
        if bits[i] and bits[j] and bits[k]:
            return False
    return True


def turan_bound(n, r):
    """Turan's theorem extremal edge count for K_r-free graphs on n vertices."""
    # ex(n, K_r) = (1 - 1/(r-1)) * n^2 / 2, for the balanced complete
    # (r-1)-partite graph. For r=3 this is floor(n^2/4).
    return math.floor((1 - 1.0 / (r - 1)) * n * n / 2)


EX_4_K3 = turan_bound(N_VERTICES, 3)
assert EX_4_K3 == 4

# Brute-force enumerate all 2^6 graphs on 4 vertices and find the ones
# that are triangle-free AND have exactly EX_4_K3 edges -- these are the
# Turan-extremal graphs for this instance.
marked_states = []
for bits in itertools.product([0, 1], repeat=N_EDGES):
    if sum(bits) == EX_4_K3 and is_triangle_free(bits):
        marked_states.append(bits)

CLASSICAL_MARKED_COUNT = len(marked_states)
CLASSICAL_MARKED_SET = {
    "".join(str(b) for b in reversed(bits))  # Qiskit bit order: q0 is rightmost
    for bits in marked_states
}

print(f"Turan bound ex(4, K3) = {EX_4_K3}")
print(f"Classical brute force: {CLASSICAL_MARKED_COUNT} marked graphs out of "
      f"{2 ** N_EDGES} on {N_VERTICES} vertices")
print(f"Marked bitstrings (Qiskit little-endian, q0 rightmost): "
      f"{sorted(CLASSICAL_MARKED_SET)}")

# Sanity: this instance's classical answer must be exactly 3 (the three
# balanced bipartitions {12|34, 13|24, 14|23} realized as K_{2,2}).
assert CLASSICAL_MARKED_COUNT == 3, (
    f"expected 3 Turan-extremal graphs on K4, got {CLASSICAL_MARKED_COUNT}"
)

# ---------------------------------------------------------------------
# 2. Genuine Grover search circuit over the 6 edge-qubits
# ---------------------------------------------------------------------

N = N_EDGES  # 6 qubits, search space size 2^6 = 64
M = CLASSICAL_MARKED_COUNT  # 3 marked items


def build_oracle(marked_bitstrings):
    """Phase-flip oracle marking exactly the given little-endian bitstrings."""
    qc = QuantumCircuit(N, name="oracle")
    for bitstring in marked_bitstrings:
        # multi-controlled Z on this exact computational basis state:
        # X-gate the 0-bits, apply multi-controlled Z, undo the X-gates.
        zero_positions = [i for i, b in enumerate(bitstring) if b == "0"]
        for i in zero_positions:
            qc.x(i)
        qc.h(N - 1)
        qc.mcx(list(range(N - 1)), N - 1)
        qc.h(N - 1)
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(n):
    qc = QuantumCircuit(n, name="diffuser")
    qc.h(range(n))
    qc.x(range(n))
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))
    return qc


oracle = build_oracle(sorted(CLASSICAL_MARKED_SET))
diffuser = build_diffuser(N)

# Optimal number of Grover iterations for search space 2^N with M marked items.
theta = math.asin(math.sqrt(M / 2 ** N))
iterations = max(1, round((math.pi / (4 * theta)) - 0.5))
print(f"Grover iterations used: {iterations}")

qc = QuantumCircuit(N, N)
qc.h(range(N))
for _ in range(iterations):
    qc.compose(oracle, inplace=True)
    qc.compose(diffuser, inplace=True)
qc.measure(range(N), range(N))

# ---------------------------------------------------------------------
# 3. Run on the ideal AerSimulator and compare to the classical answer
# ---------------------------------------------------------------------

backend = AerSimulator()
shots = 4096
result = backend.run(qc, shots=shots).result()
counts = result.get_counts()

hits = sum(c for bitstring, c in counts.items() if bitstring in CLASSICAL_MARKED_SET)
success_rate = hits / shots

print(f"Measured distribution (top 10): "
      f"{sorted(counts.items(), key=lambda kv: -kv[1])[:10]}")
print(f"Fraction of shots landing on a classically-verified Turan-extremal "
      f"graph: {success_rate:.4f}")

# Grover amplifies the marked subspace; for M=3, N=64 with the optimal
# iteration count the theoretical success probability is well over 90%.
SUCCESS_THRESHOLD = 0.85
verified = success_rate >= SUCCESS_THRESHOLD

if verified:
    print("PASS")
else:
    print("FAIL")
