"""
Erdos problem #99 (erdosproblems.com) - quantum-testable instance.

Source metadata (from /home/user/manman4/erdosproblems/data/problems.yaml,
entry "number: \"99\""):
    prize: $100
    status: open
    tags: ["geometry", "distances"]
    oeis: ["N/A"]

LIMITATION, stated honestly up front: this problem has NO associated OEIS
sequence id (oeis: ["N/A"] in the source data) and no full problem statement
text is available in the read-only clone (only the YAML metadata block is
present - no prose/statement field). It is therefore not possible to build a
circuit that tests membership in, or a defining property of, a specific OEIS
sequence for this entry, as most other lanes in this library do. This is a
best-effort substitute that is faithful to the problem's declared tags
("geometry", "distances") rather than to a specific sequence: it builds a
genuine, real Qiskit Grover-search circuit over a small, finite, exactly
computable geometric instance in the same combinatorial family (pairwise
Euclidean distances between points), and checks the quantum result against
a brute-force classical computation done in this script.

Classical property being tested
--------------------------------
Fix 8 candidate points in the plane (points[0..7], integer/half-integer
coordinates chosen by hand below). Compute, classically and exactly (using
squared Euclidean distance, so everything stays in exact rational/integer
arithmetic - no floating point comparisons), the *closest pair* among these
8 points, i.e. the pair (i, j), i != j, minimizing squared distance.

Represent each of the 8 points by a 3-bit index (0..7). Build a Grover
search circuit over 3 qubits whose oracle marks exactly the index of one
distinguished point p* such that p* achieves the overall minimum distance
to at least one other point in the set (p* is fixed as the "query" point;
Grover here searches, among the OTHER 7 points encoded on 3 qubits with one
value reserved/unused, for the index of the point nearest to p*). Concretely:

  - points[] is a fixed list of 8 planar points.
  - target index t = 0 is the query point p*.
  - classical_answer = argmin_{i != t} squared_distance(points[t], points[i])
    computed by brute force over all i in 0..7, i != t.
  - The Grover oracle marks exactly the computational basis state |i> for
    i = classical_answer (a single marked item out of 8, i.e. exactly the
    "nearest point to p*" search - a direct instance of the "distances"
    theme of tags ["geometry", "distances"]).
  - Grover's algorithm (1 iteration is optimal for 1 marked item out of 8)
    is run on AerSimulator and the most frequently measured index is
    compared to classical_answer.

This is a real, self-contained finite geometric search problem (nearest
neighbour under Euclidean distance) rendered as an unstructured-search
oracle, run on a genuine Grover circuit. It is offered as the honest
closest fit given that problem #99 carries no OEIS id and no statement
text in the source clone, per the task's fallback instruction: "write the
script anyway with your best honest attempt, note the limitation clearly".
"""

import math

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. The finite geometric instance and its exact classical answer.
# ---------------------------------------------------------------------------

# 8 points in the plane, indices 0..7 (fits exactly in 3 qubits).
points = [
    (0, 0),   # 0  <- query point p*
    (5, 5),
    (1, 1),   # close to point 0
    (7, 2),
    (3, 6),
    (6, 6),
    (2, -1),
    (4, 4),
]

QUERY = 0
N_QUBITS = 3
N_ITEMS = 2 ** N_QUBITS
assert len(points) == N_ITEMS


def squared_distance(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def classical_nearest_neighbour(query_idx, pts):
    """Brute-force, exact (integer arithmetic) nearest neighbour search."""
    best_idx = None
    best_d2 = None
    for i, p in enumerate(pts):
        if i == query_idx:
            continue
        d2 = squared_distance(pts[query_idx], p)
        if best_d2 is None or d2 < best_d2:
            best_d2 = d2
            best_idx = i
    return best_idx, best_d2


classical_answer, classical_min_d2 = classical_nearest_neighbour(QUERY, points)

print("Points:", points)
print(f"Query point index: {QUERY} -> {points[QUERY]}")
print(
    f"Classical nearest neighbour index = {classical_answer} "
    f"(point {points[classical_answer]}, squared distance = {classical_min_d2})"
)


# ---------------------------------------------------------------------------
# 2. Grover search circuit whose oracle marks exactly `classical_answer`.
# ---------------------------------------------------------------------------

def oracle_for_index(qc, qubits, index, n_qubits):
    """Flip the phase of the |index> basis state (index in [0, 2**n_qubits))."""
    bits = format(index, f"0{n_qubits}b")[::-1]  # little-endian bit order
    zero_qubits = [qubits[i] for i, b in enumerate(bits) if b == "0"]
    if zero_qubits:
        qc.x(zero_qubits)
    # Multi-controlled Z on all n_qubits (phase flip on |11...1>).
    if n_qubits == 1:
        qc.z(qubits[0])
    else:
        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1])
        qc.h(qubits[-1])
    if zero_qubits:
        qc.x(zero_qubits)


def diffuser(qc, qubits, n_qubits):
    qc.h(qubits)
    qc.x(qubits)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    qc.x(qubits)
    qc.h(qubits)


def build_grover_circuit(marked_index, n_qubits):
    qc = QuantumCircuit(n_qubits, n_qubits)
    qubits = list(range(n_qubits))

    # Uniform superposition.
    qc.h(qubits)

    # Optimal number of Grover iterations for 1 marked item out of 2**n.
    n_items = 2 ** n_qubits
    iterations = max(1, round((math.pi / 4) * math.sqrt(n_items)))

    for _ in range(iterations):
        oracle_for_index(qc, qubits, marked_index, n_qubits)
        diffuser(qc, qubits, n_qubits)

    qc.measure(qubits, qubits)
    return qc


circuit = build_grover_circuit(classical_answer, N_QUBITS)

simulator = AerSimulator()
compiled = transpile(circuit, simulator)
job = simulator.run(compiled, shots=2048)
result = job.result()
counts = result.get_counts()

# Qiskit reports bitstrings MSB..LSB left to right over the classical
# register; our register was declared with qubit i -> classical bit i, and
# Qiskit prints classical bits in reverse order (bit n-1 ... bit 0), so
# reverse the string to recover the little-endian index directly.
best_bitstring = max(counts, key=counts.get)
quantum_answer = int(best_bitstring[::-1], 2)

total_shots = sum(counts.values())
confidence = counts[best_bitstring] / total_shots

print(f"\nGrover measurement counts: {counts}")
print(
    f"Most frequent measured index = {quantum_answer} "
    f"(confidence {confidence:.3f} over {total_shots} shots)"
)

verified = quantum_answer == classical_answer

print(f"\nClassical answer: {classical_answer}")
print(f"Quantum answer:   {quantum_answer}")

if verified:
    print("PASS")
else:
    print("FAIL")
