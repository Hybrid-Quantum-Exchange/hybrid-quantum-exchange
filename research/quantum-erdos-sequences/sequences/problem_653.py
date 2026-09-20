"""
Erdos problem #653 -- quantum-testable instance
=================================================

Source metadata (data/problems.yaml, entry "number: '653'"):
    tags: ["geometry", "distances"]
    oeis: ["possible"]

LIMITATION, stated honestly up front: this problem's entry in
erdosproblems/data/problems.yaml carries no real OEIS sequence id -- the
field is the literal placeholder string "possible", not an identifier that
resolves to an OEIS A-number. There is therefore no published integer
sequence to test membership/terms against for this problem. The tags
("geometry", "distances") place it in the Erdos distinct-distances family
(the classical Erdos distinct distances problem: for n points in the plane,
what is the minimum possible number of distinct pairwise distances?).

Rather than fabricate an OEIS value, this script builds a genuine, small,
fully-classically-checkable instance of that underlying combinatorial
question and verifies a real Grover search circuit against it:

    Classical property tested:
        Fix 4 points in the plane:
            A = (0, 0), B = (2, 0), C = (1, 1), D = (9, 4)
        (A, B, C form an isosceles triangle with AC = BC = sqrt(2) and
        AB = 2, so that triple has only 2 distinct pairwise distances; D is
        placed generically so every triple involving it is scalene with 3
        distinct pairwise distances and no accidental coincidences with any
        other triple's distances. This gives a search space with a single,
        unambiguous minimizer, which is what makes the Grover amplification
        below actually demonstrate something.)
        There are C(4,3) = 4 three-point subsets:
            index 0 (bits "00"): {A, B, C}
            index 1 (bits "01"): {A, B, D}
            index 2 (bits "10"): {A, C, D}
            index 3 (bits "11"): {B, C, D}
        For each subset, compute the number of DISTINCT pairwise distances
        among its 3 points. This is computed here in Python from first
        principles (no lookup table). We then ask: which subset(s) achieve
        the minimum number of distinct distances (the local instance of the
        Erdos distinct-distances minimization)? That is a well-defined,
        finite, computable search problem over a 2-qubit index space.

    Quantum computation:
        A 2-qubit Grover search whose oracle marks exactly the index/indices
        achieving the classical minimum, run on the ideal AerSimulator, and
        compared to the classical brute-force answer.

This is a direct arithmetic/oracle circuit (Grover search over a 4-element
space), not a fabricated OEIS lookup.
"""

import itertools
import math

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical computation (first principles, in-script)
# ---------------------------------------------------------------------------

POINTS = {
    "A": (0, 0),
    "B": (2, 0),
    "C": (1, 1),
    "D": (9, 4),
}

# Fixed enumeration order of the 4 subsets of size 3, matching 2-bit index
# "index -> bits" with bit0 = qubit0 (LSB), bit1 = qubit1 (MSB).
SUBSETS = [
    ("A", "B", "C"),  # index 0 -> "00"
    ("A", "B", "D"),  # index 1 -> "01"
    ("A", "C", "D"),  # index 2 -> "10"
    ("B", "C", "D"),  # index 3 -> "11"
]


def dist(p, q):
    (x1, y1), (x2, y2) = p, q
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def distinct_distance_count(triple):
    labels = triple
    pairs = itertools.combinations(labels, 2)
    dists = set()
    for u, v in pairs:
        d = dist(POINTS[u], POINTS[v])
        # Round to avoid float-equality issues; distances here are exact
        # (1, 2, sqrt2, sqrt5), well separated, so this is safe.
        dists.add(round(d, 9))
    return len(dists)


counts = [distinct_distance_count(t) for t in SUBSETS]
classical_min = min(counts)
classical_marked = [i for i, c in enumerate(counts) if c == classical_min]

print("Subset distinct-distance counts (indices 0..3):", counts)
print("Classical minimum distinct-distance count:", classical_min)
print("Classical marked (minimizing) indices:", classical_marked)


# ---------------------------------------------------------------------------
# 2. Quantum Grover search over the 2-qubit index space {0,1,2,3}
#    Oracle marks exactly `classical_marked`.
# ---------------------------------------------------------------------------

N_QUBITS = 2  # indices 0..3


def build_oracle(marked_indices, n_qubits):
    """Phase-flip oracle marking the given basis states (little-endian)."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for idx in marked_indices:
        bits = format(idx, f"0{n_qubits}b")[::-1]  # bits[i] -> qubit i
        zero_qubits = [i for i, b in enumerate(bits) if b == "0"]
        for q in zero_qubits:
            qc.x(q)
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        for q in zero_qubits:
            qc.x(q)
    return qc


def build_diffuser(n_qubits):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def grover_iterations(n, m):
    if m <= 0 or m >= n:
        return 0
    theta = math.asin(math.sqrt(m / n))
    return max(1, round((math.pi / 4 - theta / 2) / theta))


oracle = build_oracle(classical_marked, N_QUBITS)
diffuser = build_diffuser(N_QUBITS)

n_states = 2 ** N_QUBITS
iterations = grover_iterations(n_states, len(classical_marked))

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))
for _ in range(iterations):
    qc.append(oracle.to_gate(), range(N_QUBITS))
    qc.append(diffuser.to_gate(), range(N_QUBITS))
qc.measure(range(N_QUBITS), range(N_QUBITS))

sim = AerSimulator()
tqc = transpile(qc, sim)
shots = 4096
result = sim.run(tqc, shots=shots).result()
counts_out = result.get_counts()

print("Grover iterations used:", iterations)
print("Measurement counts:", counts_out)

# Most frequently measured index(es)
best_bits = max(counts_out, key=counts_out.get)
# Qiskit reports classical bit strings MSB-first (bit_{n-1} ... bit_0)
quantum_index = int(best_bits[::-1], 2)

quantum_marked_mass = sum(
    c for bits, c in counts_out.items() if int(bits[::-1], 2) in classical_marked
)
quantum_marked_fraction = quantum_marked_mass / shots

print("Quantum most-likely index:", quantum_index)
print("Fraction of shots landing on a classically-marked index:",
      quantum_marked_fraction)


# ---------------------------------------------------------------------------
# 3. Compare and report
# ---------------------------------------------------------------------------

verified = (
    quantum_index in classical_marked
    and quantum_marked_fraction > 0.5
)

if verified:
    print("PASS: Grover search recovered the classical distinct-distance "
          "minimizer.")
else:
    print("FAIL: Grover search did not recover the classical "
          "distinct-distance minimizer.")
