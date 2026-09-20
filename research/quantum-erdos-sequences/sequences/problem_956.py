"""
Erdos problem #956 (source: manman4/erdosproblems data/problems.yaml, entry
`number: "956"`) -- HONEST LIMITATION NOTE FIRST:

  problems.yaml lists this entry as:
    prize: "no"
    informal_status: open (last_update 2025-08-31)
    oeis: ["possible"]
    tags: ["geometry", "distances", "convex"]

  "possible" is NOT a real OEIS sequence id -- it is a placeholder the
  erdosproblems.com data uses to mean "an OEIS entry may exist but is not
  linked yet." There is therefore no genuine OEIS id attached to problem 956
  to build a sequence-membership circuit from. This script does NOT fabricate
  one. Instead, per the tags actually present ("geometry", "distances",
  "convex"), it builds a genuine, self-contained finite combinatorial-geometry
  search problem in the same spirit as the family of Erdos distinct-distances
  problems (of which #956 is one, concerning distances realized by points in
  convex position), and verifies a real Grover search circuit against it.

  The property tested (fully computable classically, defined and checked from
  first principles in this script, independent of any lookup table):

    Fix n = 6 points placed at equally spaced angles on a unit circle
    (indices 0..5), so ANY subset of them is automatically in convex
    position. For each of the 2^6 = 64 subsets (encoded as a 6-bit string,
    bit i = 1 iff point i is included), compute the number of *distinct*
    pairwise Euclidean distances realized by that subset, restricted to
    subsets of size >= 3 (a genuine polygon, not a single trivial pair).
    Subsets of size < 3 are assigned a sentinel value (99) so they can never
    be selected as "minimal". Let target = the true minimum distinct-distance
    count achieved by any subset of size >= 3 -- this is realized by the two
    equilateral triangles {0,2,4} and {1,3,5} inscribed in the regular
    hexagon, each of which has only 1 distinct pairwise distance (all three
    sides equal). This script computes `target` and the exact set of marked
    (winning) 6-bit indices directly, in plain Python, with no reference to
    OEIS.

  Quantum circuit: a genuine Grover search over the 6-qubit (64-state) index
  space, with a phase oracle built from the classically precomputed marked
  set (multi-controlled-Z per marked bitstring), run on the ideal AerSimulator.
  The script measures the final state and checks that the most probable
  outcome(s) are exactly the classically-verified marked (target-achieving)
  subsets, i.e. that quantum amplitude amplification found the true answer
  to the finite combinatorial-geometry search problem.

  This is a best-effort, honestly-labeled construction: it verifies a real
  quantum Grover search against a real, from-first-principles classical
  computation motivated by problem 956's own tags, but it is NOT a canonical
  "OEIS sequence membership" test, since problem 956 carries no real OEIS id.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------------
# 1. Classical computation (first principles, no external data/lookup)
# ---------------------------------------------------------------------------

N_POINTS = 6  # 6 qubits -> 64 subsets
SENTINEL = 99


def points_on_circle(n):
    return [
        (math.cos(2 * math.pi * i / n), math.sin(2 * math.pi * i / n))
        for i in range(n)
    ]


def distinct_distance_count(subset_indices, pts):
    # Only subsets of size >= 3 are eligible: a single distance realized by
    # a pair is a trivial (and universal) case, so this restricts the search
    # to genuine polygon-shaped subsets, matching the "distances" + "convex"
    # tags on problem 956 (distinct distances realized by a convex polygon).
    if len(subset_indices) < 3:
        return SENTINEL
    dists = set()
    for a, b in itertools.combinations(subset_indices, 2):
        dx = pts[a][0] - pts[b][0]
        dy = pts[a][1] - pts[b][1]
        d = round(math.hypot(dx, dy), 9)  # round to kill float noise
        dists.add(d)
    return len(dists)


pts = points_on_circle(N_POINTS)

# subset i (0..63): bit j (0=LSB) of i set  <=> point j included
subset_counts = {}
for i in range(2 ** N_POINTS):
    idxs = [j for j in range(N_POINTS) if (i >> j) & 1]
    subset_counts[i] = distinct_distance_count(idxs, pts)

target = min(subset_counts.values())
assert target != SENTINEL, "sanity: some subset of size>=2 must exist"

marked = sorted(i for i, c in subset_counts.items() if c == target)

print(f"Classical result: minimum distinct-distance count among all subsets "
      f"of {N_POINTS} equally-spaced (convex) points = {target}")
print(f"Marked (winning) subset indices (out of 0..63): {marked}")

# ---------------------------------------------------------------------------
# 2. Quantum Grover search over the 64-state index space for `marked`
# ---------------------------------------------------------------------------

N_QUBITS = N_POINTS  # 6 qubits, 64 basis states = subset indices


def build_oracle(marked_indices, n_qubits):
    """Phase oracle: flips sign of each marked computational basis state."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for m in marked_indices:
        bits = format(m, f"0{n_qubits}b")[::-1]  # bit j -> qubit j
        zero_qubits = [j for j, b in enumerate(bits) if b == "0"]
        if zero_qubits:
            qc.x(zero_qubits)
        # multi-controlled Z on all n_qubits (phase flip |111...1>)
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
        if zero_qubits:
            qc.x(zero_qubits)
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


n_marked = len(marked)
N = 2 ** N_QUBITS
# optimal number of Grover iterations
iterations = max(1, round((math.pi / 4) * math.sqrt(N / n_marked)))

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))

oracle = build_oracle(marked, N_QUBITS)
diffuser = build_diffuser(N_QUBITS)

for _ in range(iterations):
    qc.compose(oracle, inplace=True)
    qc.compose(diffuser, inplace=True)

qc.measure(range(N_QUBITS), range(N_QUBITS))

sim = AerSimulator()
tqc = transpile(qc, sim)
shots = 4096
result = sim.run(tqc, shots=shots).result()
counts = result.get_counts()

# Qiskit bit ordering: classical register string is c[n-1]...c[0]; our
# measured qubit j went to classical bit j, so reverse the string to read
# back the same integer index convention used above.
counts_by_index = {}
for bitstring, c in counts.items():
    idx = int(bitstring[::-1], 2)
    counts_by_index[idx] = counts_by_index.get(idx, 0) + c

sorted_results = sorted(counts_by_index.items(), key=lambda kv: -kv[1])
top_k = [idx for idx, _ in sorted_results[:n_marked]]

print(f"Grover iterations used: {iterations}")
print(f"Top {n_marked} most-measured indices (quantum result): {sorted(top_k)}")
print(f"Marked indices from shots' probability mass: "
      f"{sum(counts_by_index.get(m, 0) for m in marked)}/{shots}")

# ---------------------------------------------------------------------------
# 3. Compare quantum result to classical answer
# ---------------------------------------------------------------------------

quantum_found_all_marked = set(top_k) == set(marked)
marked_probability_mass = sum(counts_by_index.get(m, 0) for m in marked) / shots

verified = quantum_found_all_marked and marked_probability_mass > 0.5

if verified:
    print("PASS: Grover search's top outcomes exactly match the classically "
          "verified minimal-distinct-distance subsets, with majority "
          "probability mass on the marked states.")
else:
    print("FAIL: quantum result did not match the classical answer.")

assert verified, "quantum Grover search did not reproduce the classical answer"
