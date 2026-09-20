"""
Erdos problem #95 — quantum-testable instance.

Source metadata (from erdosproblems/data/problems.yaml, number: "95"):
    tags: ["geometry", "convex", "distances"]
    oeis: ["possible"]

LIMITATION, stated honestly: the "oeis" field for problem #95 in the source
data is the literal placeholder string "possible", not a real OEIS sequence
id (e.g. "A123456"). There is therefore no genuine OEIS integer sequence
attached to this problem to build a "small computable property of the
sequence" from. Rather than fabricate a fake OEIS id, this script instead
builds a real, self-contained finite decision problem drawn directly from
the problem's tags (geometry / convex position / pairwise distances), which
is exactly the kind of "few distinct distances among a convex point set"
question Erdos-style geometry problems in this area concern (cf. the
Erdos distinct-distances problem family). Every classical fact below is
computed from first principles in this script, not copied from anywhere.

Chosen finite instance:
    Points: the 9 integer grid points {0,1,2} x {0,1,2} (a 3x3 grid).
    Subsets: all C(9,4) = 126 subsets of 4 distinct points.
    Property P(S): S is in strict convex position (no point of S lies
        inside or on the boundary of the triangle formed by the other
        three) AND the 6 pairwise Euclidean distances among the 4 points
        of S take on exactly 2 distinct values.

This is a finite, brute-force-computable classical property (a small
"which quadruples are 2-distance convex configurations" search), exactly
of the "small search space whose answer is a known term" kind called for.

Quantum circuit: a genuine Grover search over the 126 subsets (padded to
128 = 2^7 basis states, index space of 7 qubits). The oracle is built by
marking, with multi-controlled Z gates, precisely the basis states whose
index corresponds to a subset satisfying P(S) (computed classically first,
then compiled into the circuit — the circuit does not "know" the answer by
any other means). Grover's algorithm is run on the ideal AerSimulator, and
the measurement result is checked against the classical marked-set
computation: PASS iff the most frequently measured basis states are exactly
the (or a strict subset of the) classically-marked states.
"""

import itertools
import math

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical computation (first principles, no lookups).
# ---------------------------------------------------------------------------

GRID = [(x, y) for x in range(3) for y in range(3)]  # 9 points
assert len(GRID) == 9

SUBSETS = list(itertools.combinations(range(9), 4))
assert len(SUBSETS) == math.comb(9, 4) == 126


def cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def is_convex_position(pts):
    """True iff no point of pts lies inside/on the triangle of the other 3,
    i.e. the 4 points form a (possibly degenerate-free) convex quadrilateral
    in some order. Computed via: the point is NOT in convex position iff
    one point is inside the triangle formed by the other three."""

    def point_in_triangle(p, a, b, c):
        d1 = cross(a, b, p)
        d2 = cross(b, c, p)
        d3 = cross(c, a, p)
        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
        return not (has_neg and has_pos)  # all same sign (or zero) => inside/on

    for i in range(4):
        p = pts[i]
        others = [pts[j] for j in range(4) if j != i]
        if point_in_triangle(p, *others):
            return False
    return True


def sq_distances(pts):
    dists = set()
    for a, b in itertools.combinations(pts, 2):
        dists.add((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
    return dists


def property_P(subset_idx):
    pts = [GRID[i] for i in subset_idx]
    if not is_convex_position(pts):
        return False
    return len(sq_distances(pts)) == 2


marked_subsets = [i for i, s in enumerate(SUBSETS) if property_P(s)]

print(f"Classical search: {len(SUBSETS)} subsets of 4 points from the 3x3 grid.")
print(f"Marked (convex, exactly 2 distinct distances) count: {len(marked_subsets)}")
for i in marked_subsets:
    print(f"  index {i}: points {[GRID[j] for j in SUBSETS[i]]}")

assert 0 < len(marked_subsets) < len(SUBSETS), (
    "Need a nontrivial marked set (neither empty nor everything) for a "
    "meaningful Grover search."
)

N = 128  # 2^7, padded index space (indices 126,127 are never marked: out of range)
N_QUBITS = 7
M = len(marked_subsets)


# ---------------------------------------------------------------------------
# 2. Grover oracle: phase-flip exactly the classically-marked indices.
# ---------------------------------------------------------------------------

def apply_marking_oracle(qc, qubits, marked_indices, n_qubits):
    for idx in marked_indices:
        bits = format(idx, f"0{n_qubits}b")[::-1]  # bit i on qubits[i]
        flip_qubits = [qubits[i] for i, b in enumerate(bits) if b == "0"]
        if flip_qubits:
            qc.x(flip_qubits)
        # multi-controlled Z on all n_qubits (phase flip |11...1>)
        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1])
        qc.h(qubits[-1])
        if flip_qubits:
            qc.x(flip_qubits)


def diffuser(qc, qubits, n_qubits):
    qc.h(qubits)
    qc.x(qubits)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    qc.x(qubits)
    qc.h(qubits)


def build_grover_circuit(marked_indices, n_qubits, iterations):
    qc = QuantumCircuit(n_qubits, n_qubits)
    qubits = list(range(n_qubits))
    qc.h(qubits)
    for _ in range(iterations):
        apply_marking_oracle(qc, qubits, marked_indices, n_qubits)
        diffuser(qc, qubits, n_qubits)
    qc.measure(qubits, qubits)
    return qc


optimal_iterations = max(1, round(math.pi / 4 * math.sqrt(N / M)))
print(f"Grover iterations used: {optimal_iterations} (N={N}, M={M})")

qc = build_grover_circuit(marked_subsets, N_QUBITS, optimal_iterations)

sim = AerSimulator()
SHOTS = 4096
result = sim.run(qc, shots=SHOTS).result()
counts = result.get_counts()

# Qiskit bit order: rightmost char is qubit 0 -> index bit 0 (LSB). Reverse
# the classical-order string back to an integer index consistent with the
# oracle's bit convention above (bits[i] corresponds to qubits[i] = qubit i,
# and qc.measure(qubits, qubits) puts qubit i's outcome at classical bit i,
# which appears at string position (n_qubits-1-i) since Qiskit prints
# c[n-1]...c[0]).
def bitstring_to_index(bs, n_qubits):
    # bs is Qiskit's default order: c[n-1] c[n-2] ... c[0]
    bits = bs[::-1]  # now bits[i] = value of classical/qubit i (LSB-first)
    return sum(int(bits[i]) << i for i in range(n_qubits))


measured_indices = {bitstring_to_index(bs, N_QUBITS): c for bs, c in counts.items()}
sorted_measured = sorted(measured_indices.items(), key=lambda kv: -kv[1])

top_k = sorted_measured[: max(1, M)]
top_k_indices = {idx for idx, _ in top_k}
marked_set = set(marked_subsets)

hits_in_top = len(top_k_indices & marked_set)
prob_on_marked = sum(c for idx, c in measured_indices.items() if idx in marked_set) / SHOTS

print(f"Total probability mass on classically-marked indices: {prob_on_marked:.3f}")
print(f"Of the top {len(top_k_indices)} measured indices, {hits_in_top} are truly marked.")

# PASS iff Grover clearly amplified the marked subspace: most of the shot
# probability landed on the classically-marked indices, and the single most
# frequent measured index is itself a true match.
best_index, _ = sorted_measured[0]
quantum_result_is_marked = best_index in marked_set

verified = quantum_result_is_marked and prob_on_marked > 0.5

if verified:
    print("PASS: Grover search's top result matches a classically-verified "
          "convex, exactly-2-distinct-distance 4-point subset, and the "
          "amplified probability mass is concentrated on the true marked set.")
else:
    print("FAIL: quantum search result did not match the classical answer.")
